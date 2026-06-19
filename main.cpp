#include "config.h"

// AI 推理网关入口
// 启动流程：配置解析 → 日志初始化 → 线程池初始化 → epoll 事件循环
int main(int argc, char *argv[])
{
    Config config;
    config.parse_arg(argc, argv);

    // 将推理超时配置写入 http_conn 静态变量
    http_conn::task_timeout_sec = config.task_timeout_sec;

    WebServer server;

    // 初始化 WebServer（不再需要数据库参数）
    server.init(config.PORT, config.LOGWrite, config.OPT_LINGER,
                config.TRIGMode, config.thread_num,
                config.close_log, config.actor_model);

    // 推理配置传入
    server.m_engine_latency_ms = config.engine_latency_ms;
    server.m_batch_window_ms = config.batch_window_ms;
    server.m_max_batch_size = config.max_batch_size;

    // 日志模块
    server.log_write();

    // 业务处理线程池
    server.thread_pool();

    // epoll 触发模式
    server.trig_mode();

    // 网络监听 + epoll 初始化 + 信号注册
    server.eventListen();

    // 主循环（epoll_wait + 事件分发）
    server.eventLoop();

    return 0;
}
