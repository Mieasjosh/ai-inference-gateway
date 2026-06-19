#include "buffer_pool.h"
#include <cstdlib>
#include <cstring>

BufferPool::BufferPool(size_t buffer_size, size_t pool_size)
    : buffer_size_(buffer_size), pool_size_(pool_size)
{
    // 预分配 pool_size 个 buffer
    for (size_t i = 0; i < pool_size_; ++i) {
        float *buf = static_cast<float *>(
            std::calloc(buffer_size_, sizeof(float)));
        if (buf) {
            free_queue_.push(buf);
        }
    }
}

BufferPool::~BufferPool()
{
    // 释放池中所有 buffer
    while (!free_queue_.empty()) {
        float *buf = free_queue_.front();
        free_queue_.pop();
        std::free(buf);
    }
}

float *BufferPool::acquire()
{
    lock_.lock();
    if (free_queue_.empty()) {
        lock_.unlock();
        // 池耗尽，降级为直接 malloc
        return static_cast<float *>(
            std::calloc(buffer_size_, sizeof(float)));
    }
    float *buf = free_queue_.front();
    free_queue_.pop();
    lock_.unlock();

    // 使用前清零（复用可能有残留数据）
    std::memset(buf, 0, buffer_size_ * sizeof(float));
    return buf;
}

void BufferPool::release(float *buf)
{
    if (!buf) return;

    lock_.lock();
    if (free_queue_.size() >= pool_size_) {
        // 池已满，直接释放（说明之前有 fallback malloc 的 buf 被归还）
        lock_.unlock();
        std::free(buf);
        return;
    }
    free_queue_.push(buf);
    lock_.unlock();
}

size_t BufferPool::free_count() const
{
    const_cast<locker &>(lock_).lock();
    size_t count = free_queue_.size();
    const_cast<locker &>(lock_).unlock();
    return count;
}
