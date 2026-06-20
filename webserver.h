#ifndef WEBSERVER_H
#define WEBSERVER_H

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <cassert>
#include <sys/epoll.h>

#include "./threadpool/threadpool.h"
#include "./http/http_conn.h"
#include "./scheduler/batch_scheduler.h"
#include "./engine/mock_engine.h"
#include "./engine/onnx_engine.h"

const int MAX_FD=65536;//最大文件描述符
const int MAX_EVENT_NUMBER=10000;//epoll 内核事件表能监听的最大事件数
const int TIMESLOT=5;//最小超时单位,是心跳检测 / 连接超时的基础时间粒度

class WebServer
{
public:
    WebServer();//初始化 http_conn 数组、静态资源根目录、定时器数组。
    ~WebServer();//关闭文件描述符、释放动态分配的内存

    void init(int port, int log_write, int opt_linger, int trigmode,
              int thread_num, int close_log, int actor_model);

    void thread_pool();
    void log_write();
    void trig_mode();
    void eventListen();
    void eventLoop();
    void timer(int connfd, struct sockaddr_in client_address);
    void adjust_timer(util_timer *timer);
    void deal_timer(util_timer *timer, int sockfd);
    bool dealclientdata();
    bool dealwithsingal(bool &timeout, bool &stop_server);
    void dealwithread(int sockfd);
    void dealwithwrite(int sockfd);

public:
    int m_port;
    char *m_root;
    int m_log_write;
    int m_close_log;
    int m_actormodel;

    int m_pipefd[2];
    int m_epollfd;
    http_conn *users;

    threadpool<http_conn> *m_pool;
    int m_thread_num;

    epoll_event events[MAX_EVENT_NUMBER];
    int m_listenfd;
    int m_OPT_LINGER;
    int m_TRIGMode;
    int m_LISTENTrigmode;
    int m_CONNTrigmode;

    client_data *users_timer;
    Utils utils;

    // 推理调度相关
    int m_engine_latency_ms;         // 模拟推理延迟（仅 MockEngine 使用）
    int m_batch_window_ms;           // 批处理窗口
    int m_max_batch_size;            // 最大批大小
    int m_max_concurrent_batches;    // 最大并发 batch 数
    int m_max_queue_size;            // 调度队列最大长度

    // 引擎选择
    char *m_engine_type;             // "mock" 或 "onnx"
    char *m_model_path;              // ONNX 模型路径

    // 推理引擎（双引擎，运行时根据 engine_type 选择）
    MockEngine m_mock_engine;
    OnnxEngine m_onnx_engine;
    IInferenceEngine *m_engine;      // 指向上述两者之一

    BatchScheduler m_scheduler;      // 动态批处理调度器
};


#endif