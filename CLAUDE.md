# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

C++ 实现的 AI 模型推理服务网关。把单实例 AI 模型封装成高并发 HTTP 后端服务，核心通过**动态批处理（Dynamic Batching）**将多个推理请求合并为一次模型计算，大幅提升吞吐量。

基于 `../web_server/`（epoll + Reactor/Proactor + 线程池 + HTTP 解析 + 定时器）改造而来。

## 构建与运行

**WSL2 已安装，所有编译/测试必须通过 WSL 执行。** 当前 Windows 环境没有 g++。

从 Windows 侧操作 WSL 的固定前缀：
```bash
wsl bash -c "cd '/mnt/e/AI coding/C++/C++ai' && <你的命令>"
```

### 编译

```bash
# 编译（DEBUG=1 默认开启调试符号）
wsl bash -c "cd '/mnt/e/AI coding/C++/C++ai' && make clean && make"

# 编译优化版本
wsl bash -c "cd '/mnt/e/AI coding/C++/C++ai' && make clean && make DEBUG=0"
```

### 运行与测试

```bash
# 1. 杀掉旧实例（如果有）
wsl bash -c "kill \$(ps aux | grep ai_gateway | grep -v grep | awk '{print \$2}') 2>/dev/null"

# 2. 启动服务（后台）
wsl bash -c "cd '/mnt/e/AI coding/C++/C++ai' && nohup ./ai_gateway -p 9999 -t 8 > /tmp/ai_gateway.log 2>&1 & sleep 1"

# 3. curl 快速验证
wsl bash -c "curl -s -X POST http://127.0.0.1:9999/infer -H 'Content-Type: application/json' -d '{\"model\":\"mock\",\"input\":[1.0,2.0,3.0,4.0]}'"

# 4. Python 测试套件（端到端：单请求/顺序/并发/错误处理）
wsl bash -c "cd '/mnt/e/AI coding/C++/C++ai' && python3 test_phase3.py"

# 5. 停止服务
wsl bash -c "kill \$(ps aux | grep 'ai_gateway -p 9999' | grep -v grep | awk '{print \$2}')"
```

**注意：** WSL 中的 Python3 使用 `json.dumps` 默认在 `:` 后加空格。HTTP JSON 解析器已兼容此格式。

## 架构总览：四层模型

```
HTTP 请求 → 网络接入层 → 推理任务调度层 → 推理引擎适配层 → 资源管理层
              ↑                ↑                ↑              ↑
         webserver.*       scheduler/       engine/          memory/
         http/             batch_scheduler  engine_interface buffer_pool
         threadpool/       inference_task   mock_engine
```

### 1. 网络接入层（复用 web_server）
- **webserver**：epoll 事件循环（支持 LT/ET）、连接管理、定时器超时
- **http/http_conn**：HTTP/1.1 解析（请求行/头部/body）、`/infer` 路由、JSON 响应构造
- **threadpool**：pthread 线程池模板，worker 线程调用 `http_conn::process()` 处理请求
- **timer/lst_timer**：升序链表定时器，SIGALRM 驱动，用于连接超时
- **lock/locker**：互斥锁、信号量、条件变量封装
- **log**：同步/异步日志（block_queue 实现异步）

### 2. 推理任务调度层（核心新增，最大亮点）
- **scheduler/inference_task**：推理任务结构体，包含输入/输出/超时/同步原语（pthread mutex + cond）
- **scheduler/batch_scheduler**：动态批处理调度器，运行在独立线程。三级优先级队列 → 时间窗口攒 batch → 批量提交引擎 → 逐个唤醒 worker

### 3. 推理引擎适配层（新增，解耦设计）
- **engine/engine_interface**：抽象接口 `IInferenceEngine`，定义 `init()` / `infer_batch()` / `get_model_info()` / `shutdown()`
- **engine/mock_engine**：模拟引擎，用 sleep 模拟推理延迟，返回 `input × 2` 假结果。**用于在无模型的情况下完成调度层开发和压测**
- **engine/onnx_engine**：ONNX Runtime 引擎，封装 `Ort::Session`。支持条件编译：定义 `HAS_ONNXRUNTIME` 启用真实推理，否则降级为 stub。batch 输入在第一维拼接，输出按 batch_size 拆分
- **engine/model_manager**：多模型管理器，维护 `model_name → IInferenceEngine*` 映射，线程安全路由。不持有引擎所有权

### 4. 资源管理层（新增）
- **memory/buffer_pool**：预分配 float buffer 池，避免频繁 malloc/free
- **common/types**：`InferenceInput`、`InferenceResult`、`ModelInfo` 公共结构体

## 关键数据流（当前骨架状态）

```
main.cpp 启动流程:
  配置解析 → 日志初始化 → 线程池初始化 → epoll 事件循环

HTTP 请求处理流程:
  客户端 POST /infer (JSON body)
    → epoll_wait 检测到读事件
    → dealwithread() 将 http_conn 丢入线程池
    → worker 调用 http_conn::process()
    → process_read() 解析 HTTP → do_request() 路由到 /infer
    → 构造 JSON 响应写入 m_write_buf → process_write() 封装 HTTP 响应
    → writev() 发送响应，epoll 监听到写就绪
```

**已连线：** `/infer` → `do_request()` 解析 JSON body → 创建 `InferenceTask` → `scheduler->enqueue()` + `task.wait()` → `BatchScheduler` 攒 batch → `MockEngine::infer_batch()` → 唤醒 worker → 写 JSON 响应。

## 与原 web_server 的差异

| 移除的模块 | 原因 |
|---|---|
| CGImysql/ | 推理网关不需要 MySQL 用户认证 |
| upload/ | 不需要文件上传 |
| root/ 静态目录 | 不需要静态文件服务 |
| connection_pool 依赖 | 线程池不再持有数据库连接 |

| 保留并改造的模块 | 改动 |
|---|---|
| http_conn | `init()` 去掉了 user/passwd/sqlname 参数；`do_request()` 从文件/CGI 路由改为 `/infer` API 路由；去掉 upload 解析头 |
| webserver | `init()` 去掉了数据库参数；`sql_pool()` 移除；构造函数不再初始化 upload_dir |
| threadpool | 构造函数去掉 `connection_pool*`；`run()` 里去掉 `connectionRAII` |
| config | `sql_num` 改为 `batch_window_ms` / `max_batch_size` / `engine_latency_ms` |
| main.cpp | 去掉数据库连接字符串；不再调用 `sql_pool()` |

## 线程模型

- **主线程**：`eventLoop()` 跑 epoll_wait，分发 I/O 事件，处理 accept/信号/定时器
- **worker 线程**（线程池）：执行 `http_conn::process()`（HTTP 解析 + 业务处理），通过 EPOLLONESHOT 保证同一 fd 不被多线程同时处理
- **调度线程**（待连线）：`BatchScheduler::run()`，从优先级队列取任务、组 batch、调用引擎推理、唤醒 worker

worker 线程和调度线程之间通过 **InferenceTask 的条件变量**通信：worker 投递任务后 `task->wait()` 阻塞；调度线程推理完成后 `task->mark_done()` 唤醒。

## 当前状态与实现计划

### 已完成
- ✅ 项目骨架编译通过，生成 `ai_gateway` 二进制
- ✅ 所有新模块（engine / scheduler / memory / common）已实现
- ✅ 完整链路已验证：单请求、顺序请求、并发批处理均返回正确结果
- ✅ `http_conn::do_request()` → `BatchScheduler` → `MockEngine` 完整连线
- ✅ **阶段三完成**：`OnnxEngine`（支持条件编译 stub/真实 onnxruntime）+ `ModelManager`（多模型路由）

### 待实现
- [ ] 超时控制（复用 lst_timer）
- [ ] 优先级调度验证
- [ ] 并发限流
- [ ] 压测对比（单请求 vs 批处理 QPS）
- [ ] 下载 onnxruntime SDK 并启用 `HAS_ONNXRUNTIME` 编译宏进行真实推理测试

### 已知 warning
`lock/locker.h` 析构函数中 `throw std::exception()` 在 C++11 下触发 `-Wterminate`。不影响运行，后续统一处理。

<!-- superpowers-zh:begin (do not edit between these markers) -->
# Superpowers-ZH 中文增强版

本项目已安装 superpowers-zh 技能框架（20 个 skills）。

## 核心规则

1. **收到任务时，先检查是否有匹配的 skill** — 哪怕只有 1% 的可能性也要检查
2. **设计先于编码** — 收到功能需求时，先用 brainstorming skill 做需求分析
3. **测试先于实现** — 写代码前先写测试（TDD）
4. **验证先于完成** — 声称完成前必须运行验证命令

## 可用 Skills

Skills 位于 `.claude/skills/` 目录，每个 skill 有独立的 `SKILL.md` 文件。

- **brainstorming**: 在任何创造性工作之前必须使用此技能——创建功能、构建组件、添加功能或修改行为。在实现之前先探索用户意图、需求和设计。
- **chinese-code-review**: 中文 review 沟通参考——话术模板、分级标注（必须修复/建议修改/仅供参考）、国内团队常见反模式应对。仅在用户显式 /chinese-code-review 时调用，不要根据上下文自动触发。
- **chinese-commit-conventions**: 中文 commit 与 changelog 配置参考——Conventional Commits 中文适配、commitlint/husky/commitizen 中文模板、conventional-changelog 中文配置。仅在用户显式 /chinese-commit-conventions 时调用，不要根据上下文自动触发。
- **chinese-documentation**: 中文文档排版参考——中英文空格、全半角标点、术语保留、链接格式、中文文案排版指北约定。仅在用户显式 /chinese-documentation 时调用，不要根据上下文自动触发。
- **chinese-git-workflow**: 国内 Git 平台配置参考——Gitee、Coding.net、极狐 GitLab、CNB 的 SSH/HTTPS/凭据/CI 接入差异与镜像同步配置。仅在用户显式 /chinese-git-workflow 时调用，不要根据上下文自动触发。
- **dispatching-parallel-agents**: 当面对 2 个以上可以独立进行、无共享状态或顺序依赖的任务时使用
- **executing-plans**: 当你有一份书面实现计划需要在单独的会话中执行，并设有审查检查点时使用
- **finishing-a-development-branch**: 当实现完成、所有测试通过、需要决定如何集成工作时使用——通过提供合并、PR 或清理等结构化选项来引导开发工作的收尾
- **mcp-builder**: MCP 服务器构建方法论 — 系统化构建生产级 MCP 工具，让 AI 助手连接外部能力
- **receiving-code-review**: 收到代码审查反馈后、实施建议之前使用，尤其当反馈不明确或技术上有疑问时——需要技术严谨性和验证，而非敷衍附和或盲目执行
- **requesting-code-review**: 完成任务、实现重要功能或合并前使用，用于验证工作成果是否符合要求
- **subagent-driven-development**: 当在当前会话中执行包含独立任务的实现计划时使用
- **systematic-debugging**: 遇到任何 bug、测试失败或异常行为时使用，在提出修复方案之前执行
- **test-driven-development**: 在实现任何功能或修复 bug 时使用，在编写实现代码之前
- **using-git-worktrees**: 当需要开始与当前工作区隔离的功能开发，或在执行实现计划之前使用——通过原生工具或 git worktree 回退机制确保隔离工作区存在
- **using-superpowers**: 在开始任何对话时使用——确立如何查找和使用技能，要求在任何响应（包括澄清性问题）之前调用 Skill 工具
- **verification-before-completion**: 在宣称工作完成、已修复或测试通过之前使用，在提交或创建 PR 之前——必须运行验证命令并确认输出后才能声称成功；始终用证据支撑断言
- **workflow-runner**: 在 Claude Code / OpenClaw / Cursor 中直接运行 agency-orchestrator YAML 工作流——无需 API key，使用当前会话的 LLM 作为执行引擎。当用户提供 .yaml 工作流文件或要求多角色协作完成任务时触发。
- **writing-plans**: 当你有规格说明或需求用于多步骤任务时使用，在动手写代码之前
- **writing-skills**: 当创建新技能、编辑现有技能或在部署前验证技能是否有效时使用

## 如何使用

当任务匹配某个 skill 时，使用 `Skill` 工具加载对应 skill 并严格遵循其流程。绝不要用 Read 工具读取 SKILL.md 文件。

如果你认为哪怕只有 1% 的可能性某个 skill 适用于你正在做的事情，你必须调用该 skill 检查。
<!-- superpowers-zh:end -->
