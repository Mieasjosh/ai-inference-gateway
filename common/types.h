#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <vector>
#include <string>
#include <cstdint>

// 单条推理请求的输入
struct InferenceInput {
    std::string model_name;          // 目标模型名
    std::vector<float> data;         // 输入 tensor 数据（展平为一维）
    std::vector<int64_t> shape;      // 输入 tensor 的 shape
};

// 单条推理请求的输出
struct InferenceResult {
    std::vector<float> data;         // 输出 tensor 数据（展平为一维）
    std::vector<int64_t> shape;      // 输出 tensor 的 shape
    bool success;                    // 推理是否成功
    std::string error_msg;           // 错误信息（失败时填充）
};

// 模型元信息（从引擎获取）
struct ModelInfo {
    std::string name;
    std::vector<int64_t> input_shape;   // 输入 tensor 期望的 shape（不含 batch 维）
    std::vector<int64_t> output_shape;  // 输出 tensor 的 shape（不含 batch 维）
    size_t input_size;                  // 单条输入的 float 数量
    size_t output_size;                 // 单条输出的 float 数量
};

#endif
