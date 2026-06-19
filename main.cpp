#include "config.h"

// AI 推理网关入口
// 启动流程：配置解析 → 日志初始化 → 线程池初始化 → epoll 事件循环
int main(int argc, char *argv[])
{
    Config config;
    config.parse_arg(argc, argv);

    WebServer server;

    // 初始化 WebServer（不再需要数据库参数）
    server.init(config.PORT, config.LOGWrite, config.OPT_LINGER,
                config.TRIGMode, config.thread_num,
                config.close_log, config.actor_model);

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
