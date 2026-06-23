# AI Inference Gateway

C++ 实现的高并发 AI 模型推理服务网关——将单实例 AI 模型封装成 HTTP 后端服务，核心通过**动态批处理（Dynamic Batching）**将多个推理请求合并为一次模型计算，大幅提升吞吐量（实测 QPS 提升 **7.8 倍**）。

## 架构

```
HTTP 请求 → 网络接入层 → 推理任务调度层 → 推理引擎适配层 → 资源管理层
              ↑                ↑                ↑              ↑
         webserver.*       scheduler/       engine/          memory/
         http/             batch_scheduler  engine_interface buffer_pool
         threadpool/       inference_task   mock_engine
         timer/                             onnx_engine
                                            model_manager
```

| 层级 | 职责 | 核心组件 |
|------|------|----------|
| 网络接入层 | epoll 事件循环 + HTTP/1.1 解析 + 线程池 | `webserver`, `http_conn`, `threadpool`, `lst_timer` |
| 调度层 | 三级优先级队列 → 时间窗口攒 batch → 并发限流 → 批量推理 | `BatchScheduler`, `InferenceTask` |
| 引擎层 | 可插拔推理引擎（Mock 模拟 / ONNX Runtime 真实推理） | `IInferenceEngine`, `MockEngine`, `OnnxEngine`, `ModelManager` |
| 资源层 | 预分配 float buffer 池，减少频繁 malloc/free | `BufferPool` |

## 特性

- **动态批处理**：请求在可配置的时间窗口（`-b`，默认 10ms）内积累，合并为一次模型调用，实测 7.8x QPS 提升
- **优先级调度**：三级优先级（高/中/低），客户端通过 JSON `"priority"` 字段指定，保证重要请求优先处理
- **超时控制**：毫秒级精度，worker 侧 `pthread_cond_timedwait` + scheduler 侧 deadline 过滤
- **并发限流**：`-C` 控制最大并行 batch 数（GPU 并发上限），`-Q` 限制调度队列深度（超限返回 HTTP 503）
- **双引擎架构**：`MockEngine`（无模型依赖，开发调试用）+ `OnnxEngine`（ONNX Runtime 真实推理）
- **多模型管理**：`ModelManager` 线程安全路由 `model_name → engine`
- **Reactor/Proactor** 可切换并发模型
- **Epoll LT/ET** 触发模式可配
- **无第三方 JSON 库**：手写解析器，零外部依赖

## 构建

### 依赖

| 依赖 | 说明 |
|------|------|
| g++ (C++11) | Linux 环境（WSL2 / 原生 Linux） |
| pthread | 线程 + 同步原语 |
| [ONNX Runtime v1.19.2](https://github.com/microsoft/onnxruntime) | 仅 `-E onnx` 时需要，Linux x64 SDK ~15MB |

### 首次准备

```bash
# 1. 下载 ONNX Runtime SDK
curl -L -o onnxruntime.tgz \
  'https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz'
tar xzf onnxruntime.tgz && rm onnxruntime.tgz

# 2. 生成测试用的 ONNX 模型（需要 Python + onnx 包）
python -m pip install onnx
python -c "
import onnx
from onnx import helper, TensorProto
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4])
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4])
weight = helper.make_tensor(name='weight', data_type=TensorProto.FLOAT, dims=[4], vals=[2.0,2.0,2.0,2.0])
node = helper.make_node('Mul', inputs=['input','weight'], outputs=['output'])
graph = helper.make_graph([node], 'multiply_by_two', [input_tensor], [output_tensor], [weight])
model = helper.make_model(graph, producer_name='ai_gateway_test')
model.opset_import[0].version = 14
with open('test_model.onnx', 'wb') as f:
    f.write(model.SerializeToString())
print('test_model.onnx created')
"
```

### 编译

```bash
make               # Debug 模式（-g）
make DEBUG=0       # 优化模式（-O2）
make clean         # 清理
```

## 运行

### Mock 引擎（无需模型，y=2x 假结果）

```bash
# 启动服务
LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.19.2/lib ./ai_gateway -p 9999 -t 8

# 发送推理请求
curl -s -X POST http://127.0.0.1:9999/infer \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock","input":[1.0,2.0,3.0,4.0]}'

# 响应: {"status":"ok","output":[2.0,4.0,6.0,8.0]}
```

### ONNX 真实推理

```bash
LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.19.2/lib \
  ./ai_gateway -p 9999 -t 8 -E onnx -M ./test_model.onnx
```

### MobileNet-v2 图像分类 Demo

```bash
# Python 本地推理（推荐，无需启动 C++ 服务）
python demo_mobilenet.py cat.jpg

# 通过 C++ 网关推理
python demo_mobilenet.py --server http://127.0.0.1:9999/infer cat.jpg
```

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-p PORT` | 9006 | 服务端口 |
| `-t THREADS` | 8 | 线程池 worker 数 |
| `-s MS` | 50 | Mock 引擎模拟推理延迟（ms） |
| `-T SEC` | 30 | 推理任务超时（秒） |
| `-b MS` | 10 | 批处理积累窗口（ms） |
| `-n N` | 8 | 单批最大请求数 |
| `-C N` | 2 | 最大并发 batch 数（GPU 并发上限） |
| `-Q N` | 0 | 调度队列最大长度（0=无限制，超限返回 HTTP 503） |
| `-E TYPE` | mock | 引擎类型：`mock` / `onnx` |
| `-M PATH` | — | ONNX 模型路径（`-E onnx` 时必填） |
| `-c 0\|1` | 0 | 关闭日志（0=开启 1=关闭） |
| `-l 0\|1` | 0 | 日志写入方式（0=同步 1=异步） |
| `-m 0-3` | 0 | epoll 触发模式（0:LT+LT 1:LT+ET 2:ET+LT 3:ET+ET） |
| `-o 0\|1` | 0 | 优雅关闭连接（0=不使用 1=使用） |
| `-a 0\|1` | 0 | 并发模型（0=Proactor 1=Reactor） |

## 测试

```bash
# 端到端测试（单请求/并发/错误/超时/优先级/队列过载/并发限流/ONNX）
python3 test_phase3.py

# 压测对比（顺序 vs 批处理 QPS + 延迟分位数 + 加速比）
python3 benchmark.py
```

### 压测结果示例

| 模式 | QPS | 平均延迟 | P99 延迟 |
|------|-----|----------|----------|
| 顺序基线 | 13.8 | 72ms | 76ms |
| 批处理（batch=4） | 106.8 | 38ms | 62ms |
| **加速比** | **7.8x** | — | — |

*Mock 引擎 50ms 延迟，8 worker，batch_window=10ms*

## 项目结构

```
.
├── main.cpp              # 入口：解析参数 → 初始化 WebServer → epoll 事件循环
├── webserver.h/cpp       # 核心编排：持有 http_conn[]、线程池、引擎、调度器
├── config.h/cpp          # 配置解析（getopt）
├── Makefile              # 构建（g++ -lpthread -DHAS_ONNXRUNTIME）
├── http/
│   └── http_conn.h/cpp   # HTTP/1.1 解析 + /infer API 路由 + JSON 构造
├── scheduler/
│   ├── inference_task.h  # 推理任务结构体（优先级/超时/同步原语）
│   └── batch_scheduler.h/cpp  # 动态批处理调度器
├── engine/
│   ├── engine_interface.h    # 引擎抽象接口
│   ├── mock_engine.h/cpp     # Mock 引擎（sleep + y=2x）
│   ├── onnx_engine.h/cpp     # ONNX Runtime 引擎
│   └── model_manager.h      # 多模型管理器（线程安全）
├── memory/
│   └── buffer_pool.h/cpp    # 预分配 float buffer 池
├── common/
│   └── types.h              # InferenceInput / InferenceResult / ModelInfo
├── threadpool/
│   └── threadpool.h         # pthread 线程池模板
├── timer/
│   └── lst_timer.h          # 升序链表定时器（SIGALRM 驱动连接超时）
├── lock/
│   └── locker.h             # RAII 同步原语（mutex/sem/cond）
├── log/
│   └── log.h/cpp            # 同步/异步日志
├── test_phase3.py           # Python 端到端测试套件
├── benchmark.py             # Python 压测脚本
└── demo_mobilenet.py        # MobileNet-v2 图像分类 Demo
```

## 技术要点

### 动态批处理流程

```
客户端请求到达 → Worker 解析 JSON → 创建 InferenceTask → 入队 Scheduler
                                                              ↓
Scheduler 线程循环:                                           
  sem_timedwait 等待任务 + 定期过期清理                        
  → 积累窗口等待并发请求入队                                 
  → 高→中→低优先级出队（FIFO per 优先级）                    
  → deadline 过滤过期任务                                     
  → 等待并发 batch 槽位（-C 限流）                            
  → 提交引擎 infer_batch()                                    
  → task->mark_done() 逐个唤醒 Worker                         
```

### 线程模型

| 线程 | 职责 |
|------|------|
| 主线程 | epoll_wait 事件循环，分发 accept/read/write/signal |
| Worker 线程（-t 个） | HTTP 解析 + 业务处理，EPOLLONESHOT 保证同 fd 串行 |
| Scheduler 线程 | 优先级出队 → 攒 batch → 并发限流 → 提交引擎 |
| Exec 线程（临时） | 每个 batch 独立 pthread 执行推理，不阻塞调度循环 |

## License

MIT
