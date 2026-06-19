#include "config.h"

Config::Config()//为所有配置参数设置默认值，即使用户不传入命令行参数，服务器也能以默认配置启动。
{
    //端口号,默认9006
    PORT=9006;

    //日志写入方式，默认同步
    LOGWrite=0;

     //触发组合模式,默认listenfd LT + connfd LT
     TRIGMode=0;

     //listenfd触发模式，默认LT
     LISTENTrigmode=0;

     //connfd触发模式，默认LT
     CONNTrigmode = 0;

     //优雅关闭链接，默认不使用
     OPT_LINGER=0;

     //推理调度配置
     batch_window_ms = 10;
     max_batch_size = 8;
     engine_latency_ms = 50;
     task_timeout_sec = 30;

     //线程池内的线程数量,默认8
     thread_num=8;

     //关闭日志,默认不关闭
     close_log=0;

     //并发模型,默认是proactor
     actor_model=0;
}

void Config::parse_arg(int argc,char *argv[])
{
    int opt;
    const char *str = "p:l:m:o:s:t:c:a:b:n:T:";

    while ((opt = getopt(argc, argv, str)) != -1) {
        switch (opt) {
        case 'p': PORT = atoi(optarg); break;
        case 'l': LOGWrite = atoi(optarg); break;
        case 'm': TRIGMode = atoi(optarg); break;
        case 'o': OPT_LINGER = atoi(optarg); break;
        case 's': engine_latency_ms = atoi(optarg); break;  // 原是 sql_num
        case 't': thread_num = atoi(optarg); break;
        case 'c': close_log = atoi(optarg); break;
        case 'a': actor_model = atoi(optarg); break;
        case 'b': batch_window_ms = atoi(optarg); break;
        case 'n': max_batch_size = atoi(optarg); break;
        case 'T': task_timeout_sec = atoi(optarg); break;
        default: break;
        }
    }
}