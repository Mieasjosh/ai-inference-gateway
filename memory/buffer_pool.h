#ifndef BUFFER_POOL_H
#define BUFFER_POOL_H

#include <queue>
#include <cstddef>
#include "../lock/locker.h"

// 预分配的 float buffer 池 —— 避免每次推理都 malloc/free
//
// 用法：
//   BufferPool pool(1024, 16);  // 每个 buffer 1024 个 float，共 16 个
//   float *buf = pool.acquire();
//   // ... 使用 buf ...
//   pool.release(buf);
class BufferPool {
public:
    // buffer_size: 每个 buffer 的 float 数量
    // pool_size:  预分配的 buffer 个数
    BufferPool(size_t buffer_size, size_t pool_size);
    ~BufferPool();

    // 获取一个空闲 buffer，如果没有空闲则 fallback 到 malloc
    float *acquire();

    // 归还 buffer 到池中（必须是之前从本池获取的）
    void release(float *buf);

    size_t buffer_size() const { return buffer_size_; }
    size_t free_count() const;

private:
    size_t buffer_size_;          // 每个 buffer 的 float 数
    size_t pool_size_;            // 池容量
    std::queue<float *> free_queue_;
    locker lock_;
};

#endif
