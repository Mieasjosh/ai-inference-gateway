#ifndef ENGINE_INTERFACE_H
#define ENGINE_INTERFACE_H

#include <vector>
#include <string>
#include "../common/types.h"

// 抽象推理引擎接口
// 所有具体的推理后端（Mock / onnxruntime / llama.cpp）都实现此接口
class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;

    // 初始化：加载模型文件
    //   model_path: 模型文件的路径
    // 返回 true 表示加载成功
    virtual bool init(const std::string &model_path) = 0;

    // 批量推理：接收 batch_size 条输入，返回 batch_size 条输出
    //   batch: 一组推理输入
    // 返回：对应的推理输出（顺序与输入一致）
    // 此方法为同步阻塞调用，调度器负责在合适时机调用
    virtual std::vector<InferenceResult> infer_batch(
        const std::vector<InferenceInput> &batch) = 0;

    // 获取模型的输入/输出维度信息
    virtual ModelInfo get_model_info() const = 0;

    // 释放引擎持有的所有资源（模型权重、临时 buffer 等）
    virtual void shutdown() = 0;
};

#endif
