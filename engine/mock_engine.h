#ifndef MOCK_ENGINE_H
#define MOCK_ENGINE_H

#include <string>
#include "engine_interface.h"

// 模拟推理引擎 —— 不做真实推理，用 sleep 模拟计算耗时
// 用于开发阶段验证调度器逻辑，无需下载模型
class MockEngine : public IInferenceEngine {
public:
    MockEngine();
    ~MockEngine() override;

    // 设置模拟的推理耗时（毫秒），必须在 init() 之前调用
    void set_latency_ms(int ms) { simulated_latency_ms_ = ms; }

    // 设置模型的输入/输出大小（单条请求的 float 数量）
    void set_io_size(size_t input_size, size_t output_size);

    // --- IInferenceEngine 接口实现 ---
    bool init(const std::string &model_path) override;
    std::vector<InferenceResult> infer_batch(
        const std::vector<InferenceInput> &batch) override;
    ModelInfo get_model_info() const override;
    void shutdown() override;

private:
    int simulated_latency_ms_;
    ModelInfo model_info_;
    bool initialized_;
};

#endif
