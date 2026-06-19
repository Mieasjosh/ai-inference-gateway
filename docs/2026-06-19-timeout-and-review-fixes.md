# 2026-06-19 会话修复汇总

## 一、超时控制实现（主要功能）

### 1.1 架构

```
Worker 线程                         Scheduler 线程
───────────                         ──────────────
do_request():
  task.deadline = now + T
  enqueue(&task)
  task.wait_with_timeout_ms(T) ──→  run():
                                      sem.timedwait(10ms)
                                      cleanup_expired()     ← 清理三队列
                                      collect_batch()      ← 取任务
                                      deadline 过滤        ← spawn 前过滤
                                      execute_batch()     ← 直接推理
                                      task->mark_done()
                                    ← worker 唤醒，检查 timeout
```

超时路径两条：
- **Worker 侧**：`pthread_cond_timedwait` 超时 → 置 `timeout=true, completed=true` → 返回 false
- **Scheduler 侧**：`cleanup_expired()` / deadline 过滤 → 标记超时 → `mark_done()` 唤醒

### 1.2 改动清单

| 文件 | 改动 |
|---|---|
| `lock/locker.h` | `sem` 新增 `timedwait(int timeout_ms)` |
| `scheduler/inference_task.h` | 新增 `wait_with_timeout_ms(int timeout_ms)` |
| `scheduler/batch_scheduler.h` | 新增 `cleanup_expired()` 声明 |
| `scheduler/batch_scheduler.cpp` | `run()` 改用 `sem_timedwait()` + `cleanup_expired()` + spawn 前 deadline 过滤；`collect_batch()` 去重；`execute_batch()` 精简 |
| `config.h` / `config.cpp` | 新增 `task_timeout_sec`（默认 30s，`-T` 参数） |
| `http/http_conn.h` / `.cpp` | 新增 `static task_timeout_sec`；`do_request()` 改用 `wait_with_timeout_ms()` |
| `main.cpp` | 传递 `task_timeout_sec` 给 `http_conn` |
| `webserver.h` / `webserver.cpp` | 新增 `m_engine_latency_ms` 等成员，不再硬编码 |
| `test_phase3.py` | 新增 5 个超时测试（触发/消息/精度/回归） |

### 1.3 测试结果

```
14 PASS, 0 FAIL:
- 单请求/顺序/并发/大批量/错误处理：9 tests ✓
- 超时触发 + 消息 + 精度 + 正常配置回归：5 tests ✓
```

---

## 二、代码 Review 修复（首次 Review，5 项）

| # | 类别 | 问题 | 修复 | 文件 |
|---|---|---|---|---|
| 1 | 必须 | `bool ok` 变量声明但未使用 | 移除变量 | `http/http_conn.cpp` |
| 2 | 必须 | `cleanup_expired()` 与 `collect_batch()` 重复做过期检查 | 去掉 `collect_batch()` 中的检查，由 `cleanup_expired()` 统一处理 | `scheduler/batch_scheduler.cpp` |
| 3 | 建议 | `sem::timedwait()` 用 ms，`wait_with_timeout()` 用 sec，单位不一致 | 统一改为 ms，重命名为 `wait_with_timeout_ms()` | `scheduler/inference_task.h`、`http/http_conn.cpp` |
| 4 | 建议 | `execute_batch()` 中 deadline 检查在 exec 线程中做，应提前到 `run()` 中 | 移到 `run()` 中 spawn 前过滤，`execute_batch()` 去重 | `scheduler/batch_scheduler.cpp` |
| 5 | 建议 | 缺少超时测试覆盖 | 新增 5 个超时测试 | `test_phase3.py` |

---

## 三、骨架代码 Review 修复（阶段一/二 Review，2 项）

| # | 严重程度 | 问题 | 修复 | 文件 |
|---|---|---|---|---|
| 1 | 必须 | `json_get_string()` 用 `static char buf[256]`，多线程并发时竞态 | 改为返回 `std::string`，去掉 static buf | `http/http_conn.cpp` |
| 2 | 必须 | `cond::wait()` / `cond::timewait()` 返回 `pthread_cond_wait` 原始值（0=成功但隐式转为 false），语义倒置 | 改为 `return pthread_cond_wait(...) == 0` | `lock/locker.h` |

### 修复 #1 细节

```cpp
// 修复前（有竞态）
static char *json_get_string(char *json, const char *key) {
    static char buf[256];  // 所有 worker 线程共享
    ...
    return buf;
}

// 修复后（线程安全）
static std::string json_get_string(char *json, const char *key) {
    ...
    return std::string(start, end - start);  // 栈上构造，无竞态
}
```

调用方适配：`char *model` → `std::string model_name`，空值判断从 `model ? model : "default"` → `model_name.empty() ? "default" : model_name`。

### 修复 #2 细节

```cpp
// 修复前
bool wait(pthread_mutex_t *m) {
    int ret = pthread_cond_wait(&m_cond, m);
    return ret;  // 成功返回 0 → 隐式 bool(false)，语义倒置
}

// 修复后
bool wait(pthread_mutex_t *m) {
    return pthread_cond_wait(&m_cond, m) == 0;  // 成功 → true
}
```

调用方 `block_queue.h` 使用 `if(!m_cond.wait(...))` 进入错误路径，修复后语义正确。

---

## 四、其他

| 项 | 说明 |
|---|---|
| CLAUDE.md 更新 | 更新数据流、线程模型、已完成列表、config 差异、命令行参数 |
| 日志文件清理 | `git rm --cached 2026_06_19_Serverlog`，移除追踪 |
| 顺带清理 | `run()` 中未使用的 `window_us` 变量 |

---

## 五、已知遗留问题（本次未修复）

| 问题 | 位置 | 说明 |
|---|---|---|
| `locker` 析构 throw | `lock/locker.h:73` | `-Wterminate`，项目已知 |
| `block_queue::pop` tv_nsec | `log/block_queue.h:189` | `* 1000` 应为 `* 1000000`，函数未使用 |
| `BufferPool::free_count` const_cast | `memory/buffer_pool.cpp:63` | 应改为 `mutable locker` |
| `setnoblocking`/`addfd` 重复定义 | `http_conn.cpp` / `lst_timer.cpp` | 残留代码 |
| `enqueue()` 双重锁 | `batch_scheduler.cpp:46-49` | 可合并为一次 |
| `threadpool` 无 shutdown | `threadpool/threadpool.h:106` | 原项目设计 |
