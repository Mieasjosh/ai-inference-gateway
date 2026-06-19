#ifndef CONFIG_H
#define CONFIG_H

#include "webserver.h"

using namespace std;

class Config//完成 “默认配置初始化” 和 “命令行参数解析” 两个核心逻辑
{

public:
    Config();
    ~Config(){};

    void parse_arg(int argc,char *argv[]);

    //端口号
    int PORT;

    //日志写入方式
    int LOGWrite;

    //触发组合模式
    int TRIGMode;

    //listenfd触发模式
    int LISTENTrigmode;

    //connfd触发模式
    int CONNTrigmode;

    //优雅关闭链接
    int OPT_LINGER;

    //推理调度相关
    int batch_window_ms;     // 攒 batch 最大等待时间（毫秒）
    int max_batch_size;      // 单批最大请求数
    int engine_latency_ms;   // 模拟推理引擎延迟（毫秒）
    int task_timeout_sec;    // 推理任务超时时间（秒）

    //线程池内的线程数量
    int thread_num;

    //是否关闭日志
    int close_log;

    //并发模型选择
    int actor_model;
};

#endif

