# AI 推理网关 — 实现计划

## 上下文

基于现有 `web_server`（epoll + Reactor/Proactor + 线程池 + HTTP 解析 + 定时器），改造为高并发 AI 模型推理服务网关。

目标目录：`e:\AI coding\C++\C++ai\`（WSL 路径：`/mnt/e/AI coding/C++/C++ai/`）

## 架构总览

```
┌──────────────────────────────────────────────────┐
│  网络接入层（复用 web_server）                     │
│  epoll + HTTP + 线程池 → 接收请求，返回响应        │
├──────────────────────────────────────────────────┤
│  推理任务调度层（核心新增）                         │
│  动态批处理 / 超时控制 / 优先级 / 限流             │
├──────────────────────────────────────────────────┤
│  推理引擎适配层（新增）                            │
│  抽象接口 → Mock 引擎 / onnxruntime / llama.cpp    │
├──────────────────────────────────────────────────┤
│  资源管理层（新增）                                │
│  内存池 / 异常兜底 / 模型生命周期                  │
└──────────────────────────────────────────────────┘
```

---

## 阶段一：工程骨架搭建（0→1 先跑通）

### 1.1 从 web_server 拷贝并精简

**保留的模块（直接复制）：**
- `lock/locker.h` — 互斥锁、信号量、条件变量
- `log/` — 同步/异步日志系统
- `timer/lst_timer.h` + `lst_timer.cpp` — 定时器链表
- `http/http_conn.h` + `http/http_conn.cpp` — HTTP 解析（后续改造）
- `webserver.h` + `webserver.cpp` — epoll 主循环框架
- `threadpool/threadpool.h` — 线程池模板
- `config.h` + `config.cpp` — 配置解析
- `Makefile` — 构建文件

**删除的部分：**
- `CGImysql/` — 数据库连接池（推理网关不需要 MySQL）
- `upload/` — 文件上传（不需要）
- `root/` — 静态文件目录
- `http_conn` 中的用户认证、CGI、文件上传逻辑

### 1.2 新建目录结构

```
C++ai/
├── main.cpp                    # 入口
├── Makefile
├── config.h / config.cpp       # 改造：加推理相关配置
├── webserver.h / webserver.cpp # 改造：替换业务逻辑
├── lock/                       # 直接复用
│   └── locker.h
├── log/                        # 直接复用
│   ├── log.h / log.cpp
│   └── block_queue.h
├── timer/                      # 直接复用
│   ├── lst_timer.h / lst_timer.cpp
├── http/                       # 改造：只保留 HTTP 解析
│   ├── http_conn.h / http_conn.cpp
├── threadpool/                 # 适配改造
│   └── threadpool.h
├── scheduler/                  # 新增：调度层核心
│   ├── inference_task.h        #   推理任务定义
│   ├── batch_scheduler.h       #   动态批处理调度器
│   └── batch_scheduler.cpp
├── engine/                     # 新增：推理引擎适配
│   ├── engine_interface.h      #   抽象接口
│   ├── mock_engine.h           #   模拟引擎（最先实现）
│   ├── mock_engine.cpp
│   ├── onnx_engine.h           #   onnxruntime 封装
│   └── onnx_engine.cpp
├── memory/                     # 新增：内存管理
│   ├── buffer_pool.h           #   输入/输出 buffer 池
│   └── buffer_pool.cpp
└── common/                     # 新增：公共定义
    └── types.h                 #   统一的请求/响应结构体
```

### 1.3 Makefile 适配

- 去掉 `-lmysqlclient` 依赖
- 添加 onnxruntime 链接（阶段三才启用）
- 新增源文件编译

---

## 阶段二：核心调度层实现（最大亮点）

### 2.1 推理任务定义 (`scheduler/inference_task.h`)

```cpp
struct InferenceTask {
    uint64_t task_id;           // 唯一ID
    int client_fd;              // 对应 HTTP 连接的 fd
    std::string model_name;     // 目标模型名
    std::vector<float> input;   // 输入数据（反序列化后）
    
    // 控制和状态
    time_t arrival_time;        // 到达时间
    time_t deadline;            // 超时截止时间（绝对时间）
    int priority;               // 优先级（0=高，1=中，2=低）
    
    // 回调：推理完成后填充结果
    std::vector<float> output;
    bool completed;
    bool timeout;
    
    // 用于条件变量等待
    pthread_mutex_t mutex;
    pthread_cond_t cond;
};
```

### 2.2 动态批处理调度器 (`scheduler/batch_scheduler.h`)

核心数据结构：
```
三个优先级队列（高/中/低），每个队列按到达时间排序
```

关键参数（可配置）：
- `batch_window_ms`：攒 batch 的最大等待时间（默认 10ms）
- `max_batch_size`：单个 batch 的最大请求数（默认 8）
- `max_concurrent_batches`：同时进行的推理任务数上限（默认 2）

核心流程：
```
1. 新任务到达 → 入优先级队列
2. 检查：是否达到 max_batch_size？是 → 立即提交
3. 定时器 tick（batch_window_ms）：窗口到期 → 收集所有等待任务
4. 按优先级从高到低组 batch，提交给推理引擎
5. 推理完成 → 逐个唤醒等待的任务 → 它们各自写 HTTP 响应
```

### 2.3 调度器与事件循环的集成

改造 `webserver.cpp` 中 `http_conn::process()` 的逻辑：

**原逻辑：** 读文件 / 查数据库 / CGI → 构造响应 → 写回
**新逻辑：** 解析请求体（JSON）→ 创建 InferenceTask → 投递到 BatchScheduler → 阻塞等待 → 结果转 JSON 响应 → 写回

线程池 worker 的改动（`threadpool.h`）：
```
原：worker 调用 http_conn::process()（包含了 IO 读写 + 业务处理）
新：IO 读写仍在事件循环中，worker 只负责：
    1. 反序列化请求 → 创建任务
    2. 投递到调度器，阻塞等待结果
    3. 序列化结果 → 标记写就绪
```

### 2.4 超时控制

复用现有定时器链表机制：
- 每个 InferenceTask 注册一个 util_timer
- 超时时间 = arrival_time + deadline
- 超时回调：设置 task->timeout = true，pthread_cond_signal 唤醒等待线程
- 线程醒来检查 timeout 标志，返回 408/503 错误

### 2.5 并发限流

在 BatchScheduler 中维护：
- `active_batch_count`：当前正在推理的 batch 数
- 达到 max_concurrent_batches 时，新任务排队等待（不进 batch 窗口）

---

## 阶段三：推理引擎适配层

### 3.1 抽象接口 (`engine/engine_interface.h`)

```cpp
class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;
    
    // 初始化：加载模型
    virtual bool init(const std::string& model_path) = 0;
    
    // 批推理：输入 N 个请求的 input，返回 N 个 output
    // 纯同步调用，由调度器决定何时调用
    virtual std::vector<InferenceResult> infer_batch(
        const std::vector<InferenceInput>& batch
    ) = 0;
    
    // 获取模型输入/输出维度信息
    virtual ModelInfo get_model_info() const = 0;
    
    // 资源释放
    virtual void shutdown() = 0;
};
```

### 3.2 模拟引擎 (`engine/mock_engine.h`)

```cpp
class MockEngine : public IInferenceEngine {
    int simulated_latency_ms_;  // 模拟单 batch 推理耗时
public:
    std::vector<InferenceResult> infer_batch(...) override {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(simulated_latency_ms_)
        );
        // 返回模拟结果：input[0] * 2 之类
    }
};
```

**价值：在没有模型的情况下，完成调度层的全部开发和压测。**

### 3.3 onnxruntime 引擎 (`engine/onnx_engine.h`)

依赖：`libonnxruntime.so`（下载官方预编译包，无需编译）

```cpp
class OnnxEngine : public IInferenceEngine {
    Ort::Env env_;
    Ort::Session* session_;
    // ...
    std::vector<InferenceResult> infer_batch(...) override;
};
```

关键细节：
- batch 输入拼接：多个请求的 input tensor 在第一个维度拼接
- 输出拆分：按 batch_size 拆回每个请求的输出
- 内存复用：输入/输出 buffer 从 BufferPool 获取

### 3.4 多模型管理

在 `engine/` 层之上加一个 `ModelManager`：
```
std::map<std::string, IInferenceEngine*> models_;
```
根据请求中的 `model_name` 路由到对应的引擎实例。

---

## 阶段四：资源管理层

### 4.1 内存池 (`memory/buffer_pool.h`)

预分配一批固定大小的 buffer（如 batch_size × input_size × sizeof(float)），用队列管理：
```cpp
class BufferPool {
    std::queue<float*> free_buffers_;
    size_t buffer_size_;
    size_t pool_size_;
public:
    float* acquire();  // 获取一个 buffer
    void release(float* buf);  // 归还
};
```

避免每次推理都 `malloc` / `free`，降低延迟抖动。

### 4.2 异常兜底与降级

```
模型加载失败 → 重试 3 次，间隔 1s → 仍失败则标记模型不可用
推理执行失败 → 返回 500，释放 batch 内所有任务
推理超时（引擎层面）→ 返回 504，记录日志
内存池耗尽 → 降级为直接 malloc（不阻塞）
```

---

## 阶段五：压测与验证

### 5.1 功能验证（Mock 引擎）

```bash
# 单请求
curl -X POST http://localhost:9006/infer \
  -H "Content-Type: application/json" \
  -d '{"model":"test","input":[1.0,2.0,3.0]}'

# 并发压测（需要 Apache Bench 或 wrk）
wrk -t4 -c100 -d30s --script=post.lua http://localhost:9006/infer
```

### 5.2 对比实验

| 场景 | QPS | P50 延迟 | P99 延迟 |
|---|---|---|---|
| 直接单请求推理（batch=1） | ~20 | 50ms | 55ms |
| 动态批处理（batch_window=10ms） | ~150 | 45ms | 60ms |

**有数据对比，面试时才讲得清楚。**

### 5.3 异常场景测试

- 大量慢请求：超时控制是否生效
- 推理引擎故障：降级逻辑是否正确
- 并发打到限流阈值：限流是否保护系统

---

## 关键设计决策

1. **调度器跑在主线程的事件循环中，还是独立线程？**
   → 独立线程。通过条件变量与 worker 线程通信，避免阻塞 epoll 主循环。

2. **Worker 线程阻塞等待推理结果，会不会占满线程池？**
   → 这就是设计要点。Worker 投递任务后进入条件变量等待，释放 CPU。推理由引擎线程异步执行，完成后 broadcast 唤醒。
   线程池大小 > max_concurrent_batches × max_batch_size，确保有足够线程处理并发。

3. **优先级调度和批处理的矛盾？**
   → 高优先级的请求可以「插队」进当前正在攒的 batch，但不拆散已有 batch。如果高优请求到达时上一个 batch 刚刚提交，它必须在下一个窗口等待——这是吞吐优先的取舍。

---

## 文件改造清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `main.cpp` | 重写 | 去掉数据库相关，启动 BatchScheduler |
| `config.h/cpp` | 改造 | 新增 batch_window、max_batch_size、model_path 等配置 |
| `webserver.h/cpp` | 改造 | 去掉 CGImysql/upload，集成调度器 |
| `http/http_conn.h/cpp` | 改造 | 简化 do_request()，路由到推理处理 |
| `threadpool/threadpool.h` | 改造 | T 从 http_conn 改为 InferenceTask |
| `lock/locker.h` | 复用 | 不变 |
| `log/` | 复用 | 不变 |
| `timer/` | 复用 | 不变 |
| `scheduler/*` | 新建 | 调度层核心 |
| `engine/*` | 新建 | 推理引擎 + 适配 |
| `memory/*` | 新建 | 内存池 |
| `common/*` | 新建 | 公共类型 |

---

## 实现顺序

1. 搭骨架 — 拷贝复用模块，改 Makefile，编译通过
2. 写 MockEngine + 简单 HTTP 路由，跑通「请求→推理→响应」
3. 写 BatchScheduler + InferenceTask，实现动态批处理核心逻辑
4. 加超时控制 + 优先级 + 并发限流
5. 接入 onnxruntime，替换 MockEngine
6. 加 BufferPool + 异常兜底
7. 压测 + 优化
