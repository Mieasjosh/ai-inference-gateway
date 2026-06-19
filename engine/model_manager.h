#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include <map>
#include <string>
#include <vector>
#include "engine_interface.h"
#include "../lock/locker.h"

// 多模型管理器 —— 在引擎适配层之上做模型路由
//
// 维护 model_name → IInferenceEngine* 的映射，根据请求中的 model_name
// 将推理任务路由到对应的引擎实例。
//
// 不持有引擎所有权：调用方负责引擎的 new/delete。
// 所有方法均为线程安全。
//
// 用法示例：
//   ModelManager mgr;
//   MockEngine mock;
//   OnnxEngine onnx;
//   mock.init("mock");
//   onnx.init("resnet50.onnx");
//   mgr.register_model("mock", &mock);
//   mgr.register_model("resnet50", &onnx);
//
//   auto *engine = mgr.get_engine("resnet50");
//   auto results = engine->infer_batch(inputs);
class ModelManager {
public:
    ModelManager() {}
    ~ModelManager() {}

    // 注册一个模型引擎
    //   name:   模型名（与 InferenceInput::model_name 匹配）
    //   engine: 已初始化或未初始化的引擎指针
    // 同名覆盖：如果 name 已存在，替换旧的引擎指针
    void register_model(const std::string &name, IInferenceEngine *engine)
    {
        lock_.lock();
        models_[name] = engine;
        lock_.unlock();
    }

    // 注销一个模型引擎（不 delete，只移除映射）
    void unregister_model(const std::string &name)
    {
        lock_.lock();
        models_.erase(name);
        lock_.unlock();
    }

    // 根据模型名获取引擎指针
    // 返回 nullptr 表示未注册该模型
    IInferenceEngine *get_engine(const std::string &name)
    {
        lock_.lock();
        auto it = models_.find(name);
        IInferenceEngine *engine = (it != models_.end()) ? it->second : nullptr;
        lock_.unlock();
        return engine;
    }

    // 是否存在指定模型
    bool has_model(const std::string &name) const
    {
        const_cast<locker &>(lock_).lock();
        bool exists = (models_.find(name) != models_.end());
        const_cast<locker &>(lock_).unlock();
        return exists;
    }

    // 获取所有已注册的模型名称
    std::vector<std::string> list_models() const
    {
        std::vector<std::string> names;
        const_cast<locker &>(lock_).lock();
        for (const auto &kv : models_) {
            names.push_back(kv.first);
        }
        const_cast<locker &>(lock_).unlock();
        return names;
    }

    // 关闭所有注册的引擎
    void shutdown_all()
    {
        lock_.lock();
        for (auto &kv : models_) {
            if (kv.second) {
                kv.second->shutdown();
            }
        }
        lock_.unlock();
    }

    // 已注册的模型数量
    size_t model_count() const
    {
        const_cast<locker &>(lock_).lock();
        size_t count = models_.size();
        const_cast<locker &>(lock_).unlock();
        return count;
    }

private:
    std::map<std::string, IInferenceEngine *> models_;
    locker lock_;
};

#endif // MODEL_MANAGER_H
