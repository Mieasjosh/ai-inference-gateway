#ifndef INFERENCE_TASK_H
#define INFERENCE_TASK_H

#include <cstdint>
#include <ctime>
#include <vector>
#include <string>
#include <pthread.h>
#include "../common/types.h"

// 推理任务 —— 调度器的最小执行单元
struct InferenceTask {
    // === 静态属性（创建时填入，之后只读） ===
    uint64_t task_id;              // 全局唯一 ID
    int client_fd;                 // 对应 HTTP 连接的 fd（写响应时用）
    std::string model_name;        // 目标模型名称
    std::vector<float> input_data; // 输入数据（反序列化后的 tensor，展平）
    time_t arrival_time;           // 任务到达时间（秒）
    time_t deadline;               // 超时截止时间（绝对秒）
    int priority;                  // 0=高, 1=中, 2=低

    // === 动态属性（调度和执行过程中修改） ===
    std::vector<float> output_data;// 推理结果
    bool completed;                // 推理是否已完成（成功或失败）
    bool timeout;                  // 是否因超时被取消
    bool success;                  // 推理是否成功
    std::string error_msg;         // 错误信息

    // === 同步原语 ===
    // 工作线程投递任务后在此条件变量上等待，调度器完成推理后 signal
    pthread_mutex_t mutex;
    pthread_cond_t cond;

    InferenceTask()
        : task_id(0), client_fd(-1),
          arrival_time(0), deadline(0), priority(1),
          completed(false), timeout(false), success(false)
    {
        pthread_mutex_init(&mutex, nullptr);
        pthread_cond_init(&cond, nullptr);
    }

    ~InferenceTask()
    {
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&cond);
    }

    // 禁止拷贝（mutex/cond 不可拷贝）
    InferenceTask(const InferenceTask &) = delete;
    InferenceTask &operator=(const InferenceTask &) = delete;

    // 唤醒在 cond 上等待的线程
    void signal()
    {
        pthread_cond_signal(&cond);
    }

    // 在当前任务上阻塞等待，直到 completed 被置位
    void wait()
    {
        pthread_mutex_lock(&mutex);
        while (!completed) {
            pthread_cond_wait(&cond, &mutex);
        }
        pthread_mutex_unlock(&mutex);
    }

    // 带超时的阻塞等待，timeout_ms 毫秒
    // 返回 true=正常完成，false=超时
    bool wait_with_timeout_ms(int timeout_ms)
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += timeout_ms / 1000;
        ts.tv_nsec += (timeout_ms % 1000) * 1000000;
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec += 1;
            ts.tv_nsec -= 1000000000;
        }

        pthread_mutex_lock(&mutex);
        while (!completed) {
            int rc = pthread_cond_timedwait(&cond, &mutex, &ts);
            if (rc == ETIMEDOUT) {
                timeout = true;
                completed = true;
                pthread_mutex_unlock(&mutex);
                return false;
            }
        }
        pthread_mutex_unlock(&mutex);
        return !timeout;
    }

    // 标记完成并唤醒等待线程
    void mark_done()
    {
        pthread_mutex_lock(&mutex);
        completed = true;
        pthread_mutex_unlock(&mutex);
        pthread_cond_signal(&cond);
    }
};

#endif
