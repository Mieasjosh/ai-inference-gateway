#ifndef HTTPCONNECTION_H
#define HTTPCONNECTION_H
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/epoll.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <assert.h>
#include <sys/stat.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/uio.h>
#include <map>


#include "../lock/locker.h"
#include "../timer/lst_timer.h"
#include "../log/log.h"

class BatchScheduler;  // 前向声明

class http_conn
{
public:
    //设置读取文件的名称m_real_file大小
    static const int FILENAME_LEN=200;
    //设置读缓冲区m_read_buf大小
    static const int READ_BUFFER_SIZE=2048;
    //设置写缓冲区m_read_buf大小
    static const int WRITE_BUFFER_SIZE=1024;

    //报文的请求方法，本项目只用到GET和POST
    enum METHOD{GET=0,POST,HEAD,PUT,DELETE,TRACE,OPTIONS,CONNECT,PATH};

    //主状态机的状态
    enum CHECK_STATE
    {
        CHECK_STATE_REQUESTLINE=0,//为什么要写等于0
        CHECK_STATE_HEADER,
        CHECK_STATE_CONTENT
    };

    //报文解析的结果
    enum HTTP_CODE
    {
        NO_REQUEST,//请求不完整，需要继续读取请求报文数据
        GET_REQUEST,//获得了完整的HTTP请求
        BAD_REQUEST,//HTTP请求报文有语法错误
        NO_RESOURCE,//请求资源不存在
        FORBIDDEN_REQUEST,//FORBIDDEN_REQUEST
        FILE_REQUEST,//请求资源可以正常访问
        INTERNAL_ERROR,//服务器内部错误，该结果在主状态机逻辑switch的default下，一般不会触发
        CLOSED_CONNECTION,
        JSON_RESPONSE        // 推理网关：JSON 响应（/infer 等 API 返回）
    };

    //从状态机的状态
    enum LINE_STATUS
    {
        LINE_OK=0,
        LINE_BAD,
        LINE_OPEN
    };

public:
    http_conn() {}
    ~http_conn() {}

public:
    //初始化套接字地址，函数内部会调用私有方法init
    void init(int sockfd,const sockaddr_in &addr,char *root,int TRIGMode,int close_log);
    //关闭http连接
    void close_conn(bool read_close = true);
    void process();
    //读取浏览器端发来的全部数据
    bool read_once();
    //响应报文写入函数
    bool write();
    sockaddr_in *get_address()
    {
        return &m_address;
    }

    int timer_flag;// 标记”需触发定时器关闭连接”
    int improv;// 标记”处理完成”

private:
    void init();
    //从m_read_buf读取，并处理请求报文
    HTTP_CODE process_read();
    //向m_write_buf写入响应报文数据
    bool process_write(HTTP_CODE ret);
     //主状态机解析报文中的请求行数据
    HTTP_CODE parse_request_line(char *text);
    //主状态机解析报文中的请求头数据
    HTTP_CODE parse_headers(char *text);
    //主状态机解析报文中的请求内容
    HTTP_CODE parse_content(char *text);
    //生成响应报文
    HTTP_CODE do_request();

    //m_start_line是已经解析的字符个数
    //get_line用于将指针向后偏移，指向未处理的字符   总是忘记这两个变量代表的含义！！！
    char *get_line() {return m_read_buf+m_start_line;};

    //从状态机读取一行，分析是请求报文的哪一部分
    LINE_STATUS parse_line();

    void unmap();

    //根据响应报文格式，生成对应8个部分，以下函数均由do_request调用    目前还不知道如何调用！！！！
    bool add_response(const char *format,...);
    bool add_content(const char *content);
    bool add_status_line(int status,const char *title);
    bool add_headers(int content_length);
    bool add_content_type();
    bool add_content_length(int content_length);
    bool add_linger();
    bool add_blank_line();

public:
    static int m_epollfd;
    static int m_user_count;
    static BatchScheduler *scheduler;    // 推理任务调度器（全局唯一实例）
    static int task_timeout_sec;         // 推理任务超时时间（秒）
    int m_state; //读为0，写为1

private:
    int m_sockfd;
    sockaddr_in m_address;

    //存储读取的请求报文数据
    char m_read_buf[READ_BUFFER_SIZE]; 

    //缓冲区中m_read_buf中数据的最后一个字节的下一个位置
    long m_read_idx;
    //m_checked_idx表示从状态机在m_read_buf中读取的位置
    long m_checked_idx;
    //m_read_buf中已经解析的字符个数,m_start_line是行在buffer中的起始位置，将该位置后面的数据赋给text
    int m_start_line;

    //存储发出的响应报文数据
    char m_write_buf[WRITE_BUFFER_SIZE];
    //指示buffer中的长度
    int m_write_idx;

    //主状态机的状态
    CHECK_STATE m_check_state;
    //请求方法
    METHOD m_method;

    //以下为解析请求报文中对应的6个变量
    //存储读取文件的名称
    char m_real_file[FILENAME_LEN];
    char *m_url;
    char *m_version;
    char *m_host;
    long m_content_length;
    bool m_linger;//true表示长连接，通过解析报文头部获取

    char *m_file_address;//读取服务器上的文件地址
    struct stat m_file_stat;//stat是什么？？？？有什么用？
    struct iovec m_iv[2];//io向量机制iovec
    int m_iv_count;
    int bytes_to_send;//剩余发送字节数
    int bytes_have_send;//已发送字节数
    char *doc_root; //存网站根目录

    char m_response_body[1024]; // 推理 API 的 JSON 响应体（先写到这里，再拼 HTTP 响应）
    int m_response_status;       // HTTP 响应状态码（默认 200，过载时 503）

    int m_TRIGMode;//ET是1，LT是0
    int m_close_log;

};


#endif
