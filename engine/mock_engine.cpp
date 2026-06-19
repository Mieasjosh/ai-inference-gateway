#include "mock_engine.h"
#include <thread>
#include <chrono>
#include <cstring>

MockEngine::MockEngine()
    : simulated_latency_ms_(50),
      initialized_(false)
{
    model_info_.name = "mock";
    model_info_.input_shape = {1, 10};   // 默认 10 维输入
    model_info_.output_shape = {1, 5};   // 默认 5 维输出
    model_info_.input_size = 10;
    model_info_.output_size = 5;
}

MockEngine::~MockEngine()
{
    shutdown();
}

void MockEngine::set_io_size(size_t input_size, size_t output_size)
{
    model_info_.input_size = input_size;
    model_info_.output_size = output_size;
    model_info_.input_shape = {1, static_cast<int64_t>(input_size)};
    model_info_.output_shape = {1, static_cast<int64_t>(output_size)};
}

bool MockEngine::init(const std::string &model_path)
{
    // 模拟引擎不需要加载模型文件，直接标记初始化成功
    initialized_ = true;
    return true;
}

std::vector<InferenceResult> MockEngine::infer_batch(
    const std::vector<InferenceInput> &batch)
{
    // 1. 模拟推理延迟
    std::this_thread::sleep_for(
        std::chrono::milliseconds(simulated_latency_ms_));

    // 2. 为每个输入构造一个假输出（input 值取反作为输出）
    std::vector<InferenceResult> results;
    results.reserve(batch.size());

    for (size_t i = 0; i < batch.size(); ++i) {
        InferenceResult r;
        r.success = true;
        r.data.resize(model_info_.output_size);
        r.shape = model_info_.output_shape;

        // 简单模拟：输出 = input 前 output_size 个值的 2 倍
        for (size_t j = 0; j < model_info_.output_size; ++j) {
            float val = (j < batch[i].data.size()) ? batch[i].data[j] : 0.0f;
            r.data[j] = val * 2.0f;
        }

        results.push_back(std::move(r));
    }

    return results;
}

ModelInfo MockEngine::get_model_info() const
{
    return model_info_;
}

void MockEngine::shutdown()
{
    initialized_ = false;
}
