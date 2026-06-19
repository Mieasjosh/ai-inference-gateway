#include "onnx_engine.h"
#include "../memory/buffer_pool.h"
#include <cstring>
#include <cstdlib>

OnnxEngine::OnnxEngine()
    : initialized_(false),
      num_threads_(1),
      optimization_level_(99),  // ORT_ENABLE_ALL
      input_pool_(nullptr),
      output_pool_(nullptr)
#ifdef HAS_ONNXRUNTIME
      , input_name_(nullptr),
      output_name_(nullptr)
#endif
{
    model_info_.name = "onnx";
    model_info_.input_shape = {};
    model_info_.output_shape = {};
    model_info_.input_size = 0;
    model_info_.output_size = 0;
}

OnnxEngine::~OnnxEngine()
{
    shutdown();
}

void OnnxEngine::set_buffer_pool(BufferPool *input_pool, BufferPool *output_pool)
{
    input_pool_ = input_pool;
    output_pool_ = output_pool;
}

// ===== init =====

bool OnnxEngine::init(const std::string &model_path)
{
#ifdef HAS_ONNXRUNTIME
    try {
        // 1. 创建 ONNX Runtime 环境
        env_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ai_gateway"));

        // 2. 配置 Session 选项
        session_opts_.reset(new Ort::SessionOptions());
        session_opts_->SetIntraOpNumThreads(num_threads_);

        // 设置图优化级别
        GraphOptimizationLevel opt_level;
        switch (optimization_level_) {
            case 0:  opt_level = ORT_DISABLE_ALL; break;
            case 1:  opt_level = ORT_ENABLE_BASIC; break;
            case 2:  opt_level = ORT_ENABLE_EXTENDED; break;
            default: opt_level = ORT_ENABLE_ALL; break;
        }
        session_opts_->SetGraphOptimizationLevel(opt_level);

        // 3. 加载模型
        session_.reset(new Ort::Session(*env_, model_path.c_str(), *session_opts_));

        // 4. 创建 MemoryInfo（CPU 内存）
        memory_info_.reset(new Ort::MemoryInfo(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)));

        // 5. 创建 RunOptions
        run_opts_.reset(new Ort::RunOptions());

        // 6. 提取模型元信息（input/output name + shape）
        if (!extract_model_metadata()) {
            shutdown();
            return false;
        }

        model_info_.name = model_path;  // 用模型路径作为名称
        initialized_ = true;
        return true;

    } catch (const Ort::Exception &e) {
        // ONNX Runtime 异常：模型文件损坏、版本不匹配等
        model_info_.name = "onnx_error";
        shutdown();
        return false;
    }
#else
    // 未安装 onnxruntime SDK，init 总是失败
    (void)model_path;
    initialized_ = false;
    return false;
#endif
}

// ===== infer_batch =====

std::vector<InferenceResult> OnnxEngine::infer_batch(
    const std::vector<InferenceInput> &batch)
{
#ifdef HAS_ONNXRUNTIME
    if (!initialized_ || !session_) {
        std::vector<InferenceResult> errors;
        errors.reserve(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            InferenceResult r;
            r.success = false;
            r.error_msg = "engine not initialized";
            errors.push_back(std::move(r));
        }
        return errors;
    }

    if (batch.empty()) {
        return {};
    }

    try {
        // 1. 打包 batch 输入为 [batch_size, ...input_shape]
        std::vector<float> packed_input = pack_batch_input(batch);
        size_t batch_size = batch.size();

        // 构造输入 shape：前面插入 batch 维度
        std::vector<int64_t> input_shape = model_info_.input_shape;
        input_shape.insert(input_shape.begin(), static_cast<int64_t>(batch_size));

        // 2. 创建输入 tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            *memory_info_,
            packed_input.data(),
            packed_input.size(),
            input_shape.data(),
            input_shape.size());

        // 3. 执行推理
        const char *input_names[] = {input_name_};
        const char *output_names[] = {output_name_};
        Ort::Value *input_tensors[] = {&input_tensor};

        auto output_tensors = session_->Run(
            *run_opts_,
            input_names, input_tensors, 1,
            output_names, 1);

        // 4. 提取输出数据
        if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
            std::vector<InferenceResult> errors;
            errors.reserve(batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                InferenceResult r;
                r.success = false;
                r.error_msg = "empty or invalid output tensor";
                errors.push_back(std::move(r));
            }
            return errors;
        }

        float *output_data = output_tensors[0].GetTensorMutableData<float>();

        // 5. 拆分为每个请求的结果
        return unpack_batch_output(output_data, batch_size);

    } catch (const Ort::Exception &e) {
        // 推理执行失败
        std::vector<InferenceResult> errors;
        errors.reserve(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            InferenceResult r;
            r.success = false;
            r.error_msg = std::string("onnxruntime error: ") + e.what();
            errors.push_back(std::move(r));
        }
        return errors;
    }
#else
    // stub: 未安装 SDK 时返回错误
    std::vector<InferenceResult> errors;
    errors.reserve(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        InferenceResult r;
        r.success = false;
        r.error_msg = "OnnxEngine not available (compile with HAS_ONNXRUNTIME)";
        errors.push_back(std::move(r));
    }
    return errors;
#endif
}

// ===== get_model_info / shutdown =====

ModelInfo OnnxEngine::get_model_info() const
{
    return model_info_;
}

void OnnxEngine::shutdown()
{
#ifdef HAS_ONNXRUNTIME
    run_opts_.reset();
    memory_info_.reset();
    session_.reset();
    session_opts_.reset();
    env_.reset();
    input_name_buf_.reset();
    output_name_buf_.reset();
    input_name_ = nullptr;
    output_name_ = nullptr;
#endif
    initialized_ = false;
}

// ===== private helpers =====

std::vector<float> OnnxEngine::pack_batch_input(
    const std::vector<InferenceInput> &batch)
{
    size_t per_input = model_info_.input_size;
    size_t total = batch.size() * per_input;

    std::vector<float> packed;
    packed.reserve(total);

    for (size_t i = 0; i < batch.size(); ++i) {
        const auto &input = batch[i];
        if (input.data.size() >= per_input) {
            // 拷贝前 per_input 个 float（截断多余数据）
            packed.insert(packed.end(), input.data.begin(),
                          input.data.begin() + per_input);
        } else {
            // 数据不足，先拷贝再补零
            packed.insert(packed.end(), input.data.begin(), input.data.end());
            packed.resize(packed.size() + per_input - input.data.size(), 0.0f);
        }
    }

    return packed;
}

std::vector<InferenceResult> OnnxEngine::unpack_batch_output(
    const float *batch_output, size_t batch_size)
{
    size_t per_output = model_info_.output_size;
    std::vector<InferenceResult> results;
    results.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        InferenceResult r;
        r.success = true;
        r.data.assign(batch_output + i * per_output,
                      batch_output + (i + 1) * per_output);
        r.shape = model_info_.output_shape;
        results.push_back(std::move(r));
    }

    return results;
}

#ifdef HAS_ONNXRUNTIME
bool OnnxEngine::extract_model_metadata()
{
    if (!session_) return false;

    Ort::AllocatorWithDefaultOptions allocator;

    // ---- 输入信息 ----
    if (session_->GetInputCount() < 1) return false;
    if (session_->GetOutputCount() < 1) return false;

    // 输入名
    char *raw_name = session_->GetInputName(0, allocator);
    if (!raw_name) return false;
    size_t name_len = std::strlen(raw_name) + 1;
    input_name_buf_.reset(new char[name_len]);
    std::memcpy(input_name_buf_.get(), raw_name, name_len);
    input_name_ = input_name_buf_.get();
    allocator.Free(raw_name);

    // 输入 shape
    Ort::TypeInfo input_type = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type.GetTensorTypeAndShapeInfo();
    model_info_.input_shape = input_tensor_info.GetShape();

    // 对于动态 batch 模型，首维可能是 -1 或 "batch_size"
    // 替换为 1 作为单条默认 shape
    if (!model_info_.input_shape.empty() && model_info_.input_shape[0] < 0) {
        model_info_.input_shape[0] = 1;
    }

    // 计算单条输入的 float 数
    model_info_.input_size = 1;
    for (auto dim : model_info_.input_shape) {
        if (dim > 0) model_info_.input_size *= static_cast<size_t>(dim);
    }

    // ---- 输出信息 ----
    raw_name = session_->GetOutputName(0, allocator);
    if (!raw_name) return false;
    name_len = std::strlen(raw_name) + 1;
    output_name_buf_.reset(new char[name_len]);
    std::memcpy(output_name_buf_.get(), raw_name, name_len);
    output_name_ = output_name_buf_.get();
    allocator.Free(raw_name);

    Ort::TypeInfo output_type = session_->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type.GetTensorTypeAndShapeInfo();
    model_info_.output_shape = output_tensor_info.GetShape();

    if (!model_info_.output_shape.empty() && model_info_.output_shape[0] < 0) {
        model_info_.output_shape[0] = 1;
    }

    model_info_.output_size = 1;
    for (auto dim : model_info_.output_shape) {
        if (dim > 0) model_info_.output_size *= static_cast<size_t>(dim);
    }

    return true;
}
#endif // HAS_ONNXRUNTIME
