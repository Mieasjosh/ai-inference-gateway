#include "webserver.h"

WebServer::WebServer()
{
    m_engine_latency_ms = 50;
    m_batch_window_ms = 10;
    m_max_batch_size = 8;

    users = new http_conn[MAX_FD];

    char server_path[200];
    getcwd(server_path, 200);
    m_root = (char *)malloc(strlen(server_path) + 1);
    strcpy(m_root, server_path);

    users_timer = new client_data[MAX_FD];
}

WebServer::~WebServer()
{
    close(m_epollfd);
    close(m_listenfd);
    close(m_pipefd[0]);
    close(m_pipefd[1]);
    delete[] users;
    delete[] users_timer;
    delete m_pool;
}

void WebServer::init(int port, int log_write, int opt_linger, int trigmode,
                     int thread_num, int close_log, int actor_model)
{
    m_port = port;
    m_log_write = log_write;
    m_OPT_LINGER = opt_linger;
    m_TRIGMode = trigmode;
    m_thread_num = thread_num;
    m_close_log = close_log;
    m_actormodel = actor_model;
}

void WebServer::trig_mode()
{
    if (0 == m_TRIGMode) {
        m_LISTENTrigmode = 0;
        m_CONNTrigmode = 0;
    } else if (1 == m_TRIGMode) {
        m_LISTENTrigmode = 0;
        m_CONNTrigmode = 1;
    } else if (2 == m_TRIGMode) {
        m_LISTENTrigmode = 1;
        m_CONNTrigmode = 0;
    } else if (3 == m_TRIGMode) {
        m_LISTENTrigmode = 1;
        m_CONNTrigmode = 1;
    }
}

void WebServer::log_write()
{
    if (0 == m_close_log) {
        if (1 == m_log_write)
            Log::get_instance()->init("./Serverlog", m_close_log, 2000, 800000, 800);
        else
            Log::get_instance()->init("./Serverlog", m_close_log, 2000, 800000, 0);
    }
}

void WebServer::thread_pool()
{
    m_pool = new threadpool<http_conn>(m_actormodel, m_thread_num);

    // ===== 初始化推理引擎 + 调度器 =====
    m_engine.set_latency_ms(m_engine_latency_ms);
    m_engine.set_io_size(4, 4);        // 输入 4 个 float，输出 4 个 float
    m_engine.init("mock_model");

    m_scheduler.set_batch_window_ms(m_batch_window_ms);
    m_scheduler.set_max_batch_size(m_max_batch_size);
    m_scheduler.set_max_concurrent_batches(2);
    m_scheduler.start(&m_engine);

    // 将调度器挂到 http_conn 的静态指针上，使每个连接都能访问
    http_conn::scheduler = &m_scheduler;
}

void WebServer::eventListen()
{
    m_listenfd = socket(PF_INET, SOCK_STREAM, 0);
    assert(m_listenfd >= 0);

    if (0 == m_OPT_LINGER) {
        struct linger tmp = {0, 1};
        setsockopt(m_listenfd, SOL_SOCKET, SO_LINGER, &tmp, sizeof(tmp));
    } else {
        struct linger tmp = {1, 1};
        setsockopt(m_listenfd, SOL_SOCKET, SO_LINGER, &tmp, sizeof(tmp));
    }

    int ret = 0;
    struct sockaddr_in address;
    bzero(&address, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(m_port);

    int flag = 1;
    setsockopt(m_listenfd, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag));

    ret = bind(m_listenfd, (struct sockaddr *)&address, sizeof(address));
    assert(ret >= 0);
    ret = listen(m_listenfd, 5);
    assert(ret >= 0);

    utils.init(TIMESLOT);

    m_epollfd = epoll_create1(0);
    assert(m_epollfd != -1);

    utils.addfd(m_epollfd, m_listenfd, false, m_LISTENTrigmode);
    http_conn::m_epollfd = m_epollfd;

    ret = socketpair(AF_UNIX, SOCK_STREAM, 0, m_pipefd);
    assert(ret != -1);
    utils.setnoblocking(m_pipefd[1]);
    utils.addfd(m_epollfd, m_pipefd[0], false, 0);

    utils.addsig(SIGPIPE, SIG_IGN);
    utils.addsig(SIGALRM, utils.sig_handle, false);
    utils.addsig(SIGTERM, utils.sig_handle, false);

    alarm(TIMESLOT);

    Utils::u_pipefd = m_pipefd;
    Utils::u_epollfd = m_epollfd;
}

void WebServer::timer(int connfd, struct sockaddr_in client_address)
{
    users[connfd].init(connfd, client_address, m_root, m_CONNTrigmode, m_close_log);

    users_timer[connfd].address = client_address;
    users_timer[connfd].sockfd = connfd;
    util_timer *timer = new util_timer;
    timer->user_data = &users_timer[connfd];
    timer->cb_func = cb_func;
    time_t cur = time(NULL);
    timer->expire = cur + 3 * TIMESLOT;
    users_timer[connfd].timer = timer;
    utils.m_timer_lst.add_timer(timer);
}

void WebServer::adjust_timer(util_timer *timer)
{
    time_t cur = time(NULL);
    timer->expire = cur + 3 * TIMESLOT;
    utils.m_timer_lst.adjust_timer(timer);
    LOG_INFO("%s", "adjust timer once");
}

void WebServer::deal_timer(util_timer *timer, int sockfd)
{
    timer->cb_func(&users_timer[sockfd]);
    if (timer) {
        utils.m_timer_lst.del_timer(timer);
    }
    LOG_INFO("close fd %d", users_timer[sockfd].sockfd);
}

bool WebServer::dealclientdata()
{
    struct sockaddr_in client_address;
    socklen_t client_addrlength = sizeof(client_address);
    if (0 == m_LISTENTrigmode) {
        int connfd = accept(m_listenfd, (struct sockaddr *)&client_address,
                            &client_addrlength);
        if (connfd < 0) {
            LOG_ERROR("%s:errno is:%d", "accept error", errno);
            return false;
        }
        if (http_conn::m_user_count >= MAX_FD) {
            utils.show_error(connfd, "Internal server busy");
            LOG_ERROR("%s", "Internal server busy");
            return false;
        }
        timer(connfd, client_address);
    } else {
        while (1) {
            int connfd = accept(m_listenfd, (struct sockaddr *)&client_address,
                                &client_addrlength);
            if (connfd < 0) {
                LOG_ERROR("%s:errno is:%d", "accept error", errno);
                break;
            }
            if (http_conn::m_user_count >= MAX_FD) {
                utils.show_error(connfd, "Internal server busy");
                LOG_ERROR("%s", "Internal server busy");
                break;
            }
            timer(connfd, client_address);
        }
        return false;
    }
    return true;
}

bool WebServer::dealwithsingal(bool &timeout, bool &stop_server)
{
    int ret = 0;
    int sig;
    char singals[1024];
    ret = recv(m_pipefd[0], singals, sizeof(singals), 0);
    if (ret == -1 || ret == 0) {
        return false;
    } else {
        for (int i = 0; i < ret; i++) {
            switch (singals[i]) {
            case SIGALRM:
                timeout = true;
                break;
            case SIGTERM:
                stop_server = true;
                break;
            }
        }
    }
    return true;
}

void WebServer::dealwithread(int sockfd)
{
    util_timer *timer = users_timer[sockfd].timer;

    if (1 == m_actormodel) {  // reactor
        if (timer) adjust_timer(timer);
        m_pool->append(users + sockfd, 0);
        while (true) {
            if (1 == users[sockfd].improv) {
                if (1 == users[sockfd].timer_flag) {
                    deal_timer(timer, sockfd);
                    users[sockfd].timer_flag = 0;
                }
                users[sockfd].improv = 0;
                break;
            }
        }
    } else {  // proactor
        if (users[sockfd].read_once()) {
            m_pool->append_p(users + sockfd);
            if (timer) adjust_timer(timer);
        } else {
            deal_timer(timer, sockfd);
        }
    }
}

void WebServer::dealwithwrite(int sockfd)
{
    util_timer *timer = users_timer[sockfd].timer;

    if (1 == m_actormodel) {
        if (timer) adjust_timer(timer);
        m_pool->append(users + sockfd, 1);
        while (true) {
            if (1 == users[sockfd].improv) {
                if (1 == users[sockfd].timer_flag) {
                    deal_timer(timer, sockfd);
                    users[sockfd].timer_flag = 0;
                }
                users[sockfd].improv = 0;
                break;
            }
        }
    } else {
        if (users[sockfd].write()) {
            if (timer) adjust_timer(timer);
        } else {
            deal_timer(timer, sockfd);
        }
    }
}

void WebServer::eventLoop()
{
    bool timeout = false;
    bool stop_server = false;

    while (!stop_server) {
        int number = epoll_wait(m_epollfd, events, MAX_EVENT_NUMBER, -1);
        if (number < 0 && errno != EINTR) {
            LOG_ERROR("%s", "epoll failure");
            break;
        }

        for (int i = 0; i < number; i++) {
            int sockfd = events[i].data.fd;

            if (sockfd == m_listenfd) {
                bool flag = dealclientdata();
                if (false == flag) continue;
            } else if (events[i].events & (EPOLLRDHUP | EPOLLHUP | EPOLLERR)) {
                util_timer *timer = users_timer[sockfd].timer;
                deal_timer(timer, sockfd);
            } else if ((sockfd == m_pipefd[0]) && (events[i].events & EPOLLIN)) {
                bool flag = dealwithsingal(timeout, stop_server);
                if (false == flag)
                    LOG_ERROR("%s", "dealclientdata failure");
            } else if (events[i].events & EPOLLIN) {
                dealwithread(sockfd);
            } else if (events[i].events & EPOLLOUT) {
                dealwithwrite(sockfd);
            }
        }
        if (timeout) {
            utils.timer_handler();
            LOG_INFO("%s", "timer tick");
            timeout = false;
        }
    }
}
