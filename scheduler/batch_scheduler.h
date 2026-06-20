#ifndef BATCH_SCHEDULER_H
#define BATCH_SCHEDULER_H

#include <queue>
#include <vector>
#include <deque>
#include <pthread.h>
#include <cstdint>
#include <ctime>

#include "inference_task.h"
#include "../engine/engine_interface.h"
#include "../lock/locker.h"

// 动态批处理调度器 —— 整个推理网关的核心
//
// 职责：
// 1. 接收外部投递的 InferenceTask，按优先级入队
// 2. 在一个时间窗口内攒 batch，到上限或窗口到期时提交给推理引擎
// 3. 推理完成后唤醒各任务对应的等待线程
//
// 调度器运行在独立线程中，通过条件变量与 worker 线程通信。
class BatchScheduler {
public:
    BatchScheduler();
    ~BatchScheduler();

    // === 配置参数（必须在 start() 之前设置） ===
    void set_batch_window_ms(int ms)   { batch_window_ms_ = ms; }
    void set_max_batch_size(int n)     { max_batch_size_ = n; }
    void set_max_concurrent_batches(int n);
    void set_max_queue_size(int n)     { max_queue_size_ = n; }

    // === 生命周期 ===

    // 绑定一个推理引擎实例并启动调度线程
    // 调用后调度器开始轮询队列
    bool start(IInferenceEngine *engine);

    // 停止调度线程，等待所有正在执行的 batch 完成
    void stop();

    // === 任务投递 ===

    // 提交一个推理任务到调度队列
    // 调用线程（worker）随后在 task->wait() 上阻塞，等待推理完成
    // 返回 false 表示队列已满，调用方应返回 503
    bool enqueue(InferenceTask *task);

    // 返回当前还在排队的任务数（不含正在推理的）
    int pending_count() const;

    // 是否正在运行
    bool running() const { return running_; }

private:
    // 调度线程的主循环
    static void *scheduler_thread(void *arg);
    void run();

    // 扫描三个优先级队列，标记并移除所有 deadline < now 的任务
    void cleanup_expired();

    // 从各优先级队列中取任务组成一个 batch
    // 返回 nullptr 表示当前没有待处理任务
    std::vector<InferenceTask *> collect_batch();

    // 将一个 batch 提交给推理引擎
    void execute_batch(const std::vector<InferenceTask *> &batch);

    // === 成员变量 ===

    // 三个优先级队列（高/中/低），用 deque + 简单锁保护
    // 任务按到达时间自然排序（deque::push_back 保持 FIFO）
    std::deque<InferenceTask *> queue_high_;
    std::deque<InferenceTask *> queue_mid_;
    std::deque<InferenceTask *> queue_low_;

    mutable locker queue_lock_;       // 保护三个队列
    sem queue_sem_;                   // 信号量：队列中的任务数（用于唤醒调度线程）

    // 配置参数
    int batch_window_ms_;             // 攒 batch 的最大等待时间
    int max_batch_size_;              // 单个 batch 最大任务数
    int max_concurrent_batches_;      // 同时推理的 batch 数上限
    int max_queue_size_;              // 调度队列最大长度（0=无限制）

    int active_batch_count_;          // 当前正在推理的 batch 数量

    IInferenceEngine *engine_;        // 推理引擎（不持有所有权）
    pthread_t scheduler_thread_id_;   // 调度线程
    bool running_;                    // 调度线程是否运行中

    uint64_t next_task_id_;           // 自增任务 ID
};

#endif
