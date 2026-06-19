#include "batch_scheduler.h"
#include <unistd.h>
#include <algorithm>

BatchScheduler::BatchScheduler()
    : batch_window_ms_(10),
      max_batch_size_(8),
      max_concurrent_batches_(2),
      active_batch_count_(0),
      engine_(nullptr),
      running_(false),
      next_task_id_(1)
{
}

BatchScheduler::~BatchScheduler()
{
    stop();
}

bool BatchScheduler::start(IInferenceEngine *engine)
{
    if (!engine) return false;
    engine_ = engine;
    running_ = true;

    if (pthread_create(&scheduler_thread_id_, nullptr,
                       scheduler_thread, this) != 0) {
        running_ = false;
        return false;
    }
    pthread_detach(scheduler_thread_id_);
    return true;
}

void BatchScheduler::stop()
{
    running_ = false;
    queue_sem_.post();  // 唤醒调度线程让其退出
    // 简单等待：生产环境中应等待线程实际退出
}

void BatchScheduler::enqueue(InferenceTask *task)
{
    // 分配唯一 ID
    queue_lock_.lock();
    task->task_id = next_task_id_++;
    task->arrival_time = time(nullptr);
    queue_lock_.unlock();

    // 根据优先级入队
    queue_lock_.lock();
    switch (task->priority) {
        case 0: queue_high_.push_back(task); break;
        case 2: queue_low_.push_back(task); break;
        default: queue_mid_.push_back(task); break;
    }
    queue_lock_.unlock();

    // 通知调度线程有新任务
    queue_sem_.post();
}

int BatchScheduler::pending_count() const
{
    queue_lock_.lock();
    int count = queue_high_.size() + queue_mid_.size() + queue_low_.size();
    queue_lock_.unlock();
    return count;
}

// === 调度线程入口 ===

void *BatchScheduler::scheduler_thread(void *arg)
{
    BatchScheduler *scheduler = static_cast<BatchScheduler *>(arg);
    scheduler->run();
    return nullptr;
}

void BatchScheduler::run()
{
    // batch_window 对应的微秒
    int window_us = batch_window_ms_ * 1000;

    while (running_) {
        // 等待第一个任务到达（带超时：没有任务时不消耗 CPU）
        // 如果有正在执行的 batch，减小等待时间以允许快速响应
        int wait_ms = (active_batch_count_ > 0) ? batch_window_ms_ : 500;
        queue_sem_.wait();  // TODO: 改用 timedwait，目前用 wait 简化

        if (!running_) break;

        std::vector<InferenceTask *> batch = collect_batch();
        if (!batch.empty()) {
            // 等待直到可以提交新 batch（并发限流）
            while (running_ && active_batch_count_ >= max_concurrent_batches_) {
                usleep(1000);  // 1ms 轮询等待
            }

            if (!running_) {
                // 退出前释放这批任务
                for (auto *t : batch) {
                    t->timeout = true;
                    t->mark_done();
                }
                break;
            }

            // 提交 batch 给推理引擎（在新线程中执行，避免阻塞调度循环）
            queue_lock_.lock();
            ++active_batch_count_;
            queue_lock_.unlock();

            // 创建临时线程执行推理（TODO: 改为从线程池取线程）
            std::vector<InferenceTask *> *batch_copy =
                new std::vector<InferenceTask *>(batch);
            pthread_t exec_thread;
            pthread_create(&exec_thread, nullptr,
                [](void *arg) -> void * {
                    auto *pair = static_cast<
                        std::pair<BatchScheduler *, std::vector<InferenceTask *> *> *>(arg);
                    pair->first->execute_batch(*pair->second);
                    delete pair->second;
                    delete pair;
                    return nullptr;
                },
                new std::pair<BatchScheduler *, std::vector<InferenceTask *> *>(
                    this, batch_copy));
            pthread_detach(exec_thread);
        }
    }
}

std::vector<InferenceTask *> BatchScheduler::collect_batch()
{
    std::vector<InferenceTask *> batch;

    queue_lock_.lock();

    // 从高到低优先级依次取任务
    auto collect_from = [&](std::deque<InferenceTask *> &q) {
        while (!q.empty() && static_cast<int>(batch.size()) < max_batch_size_) {
            batch.push_back(q.front());
            q.pop_front();
        }
    };

    collect_from(queue_high_);
    collect_from(queue_mid_);
    collect_from(queue_low_);

    queue_lock_.unlock();

    return batch;
}

void BatchScheduler::execute_batch(const std::vector<InferenceTask *> &batch)
{
    if (!engine_ || batch.empty()) {
        queue_lock_.lock();
        --active_batch_count_;
        queue_lock_.unlock();
        return;
    }

    // 1. 将 InferenceTask 的输入转换为 InferenceInput
    std::vector<InferenceInput> inputs;
    inputs.reserve(batch.size());

    for (auto *task : batch) {
        InferenceInput input;
        input.model_name = task->model_name;
        input.data = task->input_data;
        inputs.push_back(std::move(input));
    }

    // 2. 调用推理引擎
    std::vector<InferenceResult> results = engine_->infer_batch(inputs);

    // 3. 将结果写回各个 task 并唤醒等待线程
    for (size_t i = 0; i < batch.size(); ++i) {
        InferenceTask *task = batch[i];

        if (i < results.size() && results[i].success) {
            task->output_data = results[i].data;
            task->success = true;
        } else {
            task->success = false;
            if (i < results.size())
                task->error_msg = results[i].error_msg;
            else
                task->error_msg = "batch result mismatch";
        }

        task->mark_done();  // 唤醒等待的 worker 线程
    }

    queue_lock_.lock();
    --active_batch_count_;
    queue_lock_.unlock();
}
