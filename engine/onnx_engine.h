#ifndef ONNX_ENGINE_H
#define ONNX_ENGINE_H

#include "engine_interface.h"
#include <string>
#include <vector>
#include <memory>

// 前置声明 —— 避免强制依赖 onnxruntime 头文件
// 当 HAS_ONNXRUNTIME 未定义时，OnnxEngine 降级为 stub（init 返回 false）
#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

class BufferPool;

// ONNX Runtime 推理引擎
//
// 封装 onnxruntime C++ API，实现 IInferenceEngine 接口。
// 通过编译宏 HAS_ONNXRUNTIME 控制是否启用真实推理：
//   - 定义时：完整实现，加载 .onnx 模型并推理
//   - 未定义时：stub，init() 返回 false，infer_batch() 返回错误
//
// 用法示例：
//   OnnxEngine engine;
//   engine.set_buffer_pool(&input_pool, &output_pool);
//   engine.init("resnet50.onnx");
//   auto results = engine.infer_batch(inputs);
class OnnxEngine : public IInferenceEngine {
public:
    OnnxEngine();
    ~OnnxEngine() override;

    // 设置线程数（必须在 init() 之前调用）
    void set_num_threads(int n) { num_threads_ = n; }

    // 设置图形优化级别（必须在 init() 之前调用）
    // 0=disable, 1=basic, 2=extended, 99=all (默认)
    void set_optimization_level(int level) { optimization_level_ = level; }

    // 注入 BufferPool 用于输入/输出 tensor 分配
    // 传入 nullptr 表示不使用池，直接 malloc（默认）
    void set_buffer_pool(BufferPool *input_pool, BufferPool *output_pool);

    // --- IInferenceEngine 接口实现 ---
    bool init(const std::string &model_path) override;
    std::vector<InferenceResult> infer_batch(
        const std::vector<InferenceInput> &batch) override;
    ModelInfo get_model_info() const override;
    void shutdown() override;

private:
    // 将 batch 条输入拼接为 [batch_size, input_size] 的连续 float 数组
    // 返回的 vector 可直接用于 Ort::Value::CreateTensor
    std::vector<float> pack_batch_input(
        const std::vector<InferenceInput> &batch);

    // 将 [batch_size, output_size] 的输出拆回 batch_size 条 InferenceResult
    std::vector<InferenceResult> unpack_batch_output(
        const float *batch_output, size_t batch_size);

    // 从 Ort::Session 中提取输入/输出的 name + shape，填充 model_info_
    bool extract_model_metadata();

    ModelInfo model_info_;
    bool initialized_;
    bool has_batch_dim_;           // 模型是否已有 batch 维度（如 [1,3,224,224]）

    // 配置参数
    int num_threads_;
    int optimization_level_;

    // BufferPool（不持有所有权）
    BufferPool *input_pool_;
    BufferPool *output_pool_;

#ifdef HAS_ONNXRUNTIME
    // ONNX Runtime 核心对象（持有所有权）
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_opts_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    std::unique_ptr<Ort::RunOptions> run_opts_;

    // 输入/输出节点名（C 字符串副本，independent lifetime）
    std::unique_ptr<char[]> input_name_buf_;
    std::unique_ptr<char[]> output_name_buf_;
    const char *input_name_;   // 指向 input_name_buf_.get()
    const char *output_name_;  // 指向 output_name_buf_.get()
#endif
};

#endif // ONNX_ENGINE_H
