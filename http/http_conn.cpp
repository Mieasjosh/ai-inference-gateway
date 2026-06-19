#include "http_conn.h"
#include "../scheduler/inference_task.h"
#include "../scheduler/batch_scheduler.h"

#include <string>
#include <cstdlib>
#include <cstring>

// HTTP 响应状态文本
const char *ok_200_title = "OK";
const char *error_400_title = "Bad Request";
const char *error_400_form = "Your request has bad syntax or is inherently impossible to staisfy.\n";
const char *error_403_title = "Forbidden";
const char *error_403_form = "You do not have permission to get file form this server.\n";
const char *error_404_title = "Not Found";
const char *error_404_form = "The requested file was not found on this server.\n";
const char *error_500_title = "Internal Error";
const char *error_500_form = "There was an unusual problem serving the request file.\n";

int http_conn::m_user_count = 0;
int http_conn::m_epollfd = -1;
BatchScheduler *http_conn::scheduler = nullptr;

// ===== 简易 JSON 解析（无第三方依赖） =====

// 从 JSON 字符串中提取字符串值：搜 "\"key\":\"" → 返回 value
static char *json_get_string(char *json, const char *key)
{
    static char buf[256];
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":\"", key);
    char *start = strstr(json, search);
    if (!start) return nullptr;
    start += strlen(search);
    char *end = strchr(start, '"');
    if (!end) return nullptr;
    size_t len = end - start;
    if (len > sizeof(buf) - 1) len = sizeof(buf) - 1;
    memcpy(buf, start, len);
    buf[len] = '\0';
    return buf;
}

// 从 JSON 字符串中提取浮点数组：搜 "\"key\":[" → 解析到 vector
static std::vector<float> json_get_float_array(char *json, const char *key)
{
    std::vector<float> result;
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":[", key);
    char *start = strstr(json, search);
    if (!start) return result;
    start += strlen(search);

    // 逐个解析逗号分隔的浮点数，直到遇到 ]
    while (*start && *start != ']') {
        while (*start == ' ' || *start == ',') ++start;
        if (*start == ']' || *start == '\0') break;
        char *end;
        float val = strtof(start, &end);
        if (end == start) break;  // 解析失败
        result.push_back(val);
        start = end;
    }
    return result;
}

// ===== 工具函数（从原 web_server 保留） =====

int setnoblocking(int fd)
{
    int old_option = fcntl(fd, F_GETFL);
    int new_option = old_option | O_NONBLOCK;
    fcntl(fd, F_SETFL, new_option);
    return old_option;
}

void addfd(int epollfd, int fd, bool one_shot, int TRIGMode)
{
    epoll_event event;
    event.data.fd = fd;
    if (1 == TRIGMode)
        event.events = EPOLLIN | EPOLLET | EPOLLRDHUP;
    else
        event.events = EPOLLIN | EPOLLRDHUP;
    if (one_shot)
        event.events |= EPOLLONESHOT;
    epoll_ctl(epollfd, EPOLL_CTL_ADD, fd, &event);
    setnoblocking(fd);
}

void removefd(int epollfd, int fd)
{
    epoll_ctl(epollfd, EPOLL_CTL_DEL, fd, 0);
    close(fd);
}

void modfd(int epollfd, int fd, int ev, int TRIGMode)
{
    epoll_event event;
    event.data.fd = fd;
    if (1 == TRIGMode)
        event.events = ev | EPOLLET | EPOLLONESHOT | EPOLLRDHUP;
    else
        event.events = ev | EPOLLONESHOT | EPOLLRDHUP;
    epoll_ctl(epollfd, EPOLL_CTL_MOD, fd, &event);
}

// ===== http_conn 成员函数 =====

void http_conn::close_conn(bool real_close)
{
    if (real_close && (m_sockfd != -1)) {
        removefd(m_epollfd, m_sockfd);
        m_sockfd = -1;
        m_user_count--;
    }
}

void http_conn::init(int sockfd, const sockaddr_in &addr, char *root,
                     int TRIGMode, int close_log)
{
    m_sockfd = sockfd;
    m_address = addr;
    addfd(m_epollfd, sockfd, true, TRIGMode);
    m_user_count++;

    doc_root = root;
    m_TRIGMode = TRIGMode;
    m_close_log = close_log;

    init();
}

void http_conn::init()
{
    bytes_to_send = 0;
    bytes_have_send = 0;
    m_check_state = CHECK_STATE_REQUESTLINE;
    m_linger = false;
    m_method = GET;
    m_url = 0;
    m_version = 0;
    m_content_length = 0;
    m_host = 0;
    m_start_line = 0;
    m_checked_idx = 0;
    m_read_idx = 0;
    m_write_idx = 0;
    m_state = 0;
    timer_flag = 0;
    improv = 0;

    memset(m_read_buf, '\0', READ_BUFFER_SIZE);
    memset(m_write_buf, '\0', WRITE_BUFFER_SIZE);
    memset(m_real_file, '\0', FILENAME_LEN);
    memset(m_response_body, '\0', sizeof(m_response_body));
}

// ===== HTTP 解析（从原 web_server 保留） =====

http_conn::LINE_STATUS http_conn::parse_line()
{
    char temp;
    for (; m_checked_idx < m_read_idx; ++m_checked_idx) {
        temp = m_read_buf[m_checked_idx];
        if (temp == '\r') {
            if ((m_checked_idx + 1) == m_read_idx)
                return LINE_OPEN;
            else if (m_read_buf[m_checked_idx + 1] == '\n') {
                m_read_buf[m_checked_idx++] = '\0';
                m_read_buf[m_checked_idx++] = '\0';
                return LINE_OK;
            }
            return LINE_BAD;
        } else if (temp == '\n') {
            if (m_checked_idx > 1 && m_read_buf[m_checked_idx - 1] == '\r') {
                m_read_buf[m_checked_idx - 1] = '\0';
                m_read_buf[m_checked_idx++] = '\0';
                return LINE_OK;
            }
            return LINE_BAD;
        }
    }
    return LINE_OPEN;
}

bool http_conn::read_once()
{
    if (m_read_idx >= READ_BUFFER_SIZE)
        return false;
    int bytes_read = 0;

    if (0 == m_TRIGMode) {  // LT
        bytes_read = recv(m_sockfd, m_read_buf + m_read_idx,
                          READ_BUFFER_SIZE - m_read_idx, 0);
        m_read_idx += bytes_read;
        if (bytes_read <= 0) return false;
        return true;
    } else {  // ET
        while (true) {
            bytes_read = recv(m_sockfd, m_read_buf + m_read_idx,
                              READ_BUFFER_SIZE - m_read_idx, 0);
            if (bytes_read == -1) {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                    break;
                return false;
            } else if (bytes_read == 0) {
                return false;
            }
            m_read_idx += bytes_read;
        }
        return true;
    }
}

http_conn::HTTP_CODE http_conn::parse_request_line(char *text)
{
    m_url = strpbrk(text, " \t");
    if (!m_url) return BAD_REQUEST;
    *m_url++ = '\0';

    char *method = text;
    if (strcasecmp(method, "GET") == 0)
        m_method = GET;
    else if (strcasecmp(method, "POST") == 0)
        m_method = POST;
    else
        return BAD_REQUEST;

    m_url += strspn(m_url, " \t");
    m_version = strpbrk(m_url, " \t");
    if (!m_version) return BAD_REQUEST;
    *m_version++ = '\0';
    m_version += strspn(m_version, " \t");

    if (strcasecmp(m_version, "HTTP/1.1") != 0)
        return BAD_REQUEST;

    if (strncasecmp(m_url, "http://", 7) == 0) {
        m_url += 7;
        m_url = strchr(m_url, '/');
    }
    if (strncasecmp(m_url, "https://", 8) == 0) {
        m_url += 8;
        m_url = strchr(m_url, '/');
    }
    if (!m_url || m_url[0] != '/')
        return BAD_REQUEST;

    m_check_state = CHECK_STATE_HEADER;
    return NO_REQUEST;
}

http_conn::HTTP_CODE http_conn::parse_headers(char *text)
{
    if (text[0] == '\0') {
        if (m_content_length != 0) {
            m_check_state = CHECK_STATE_CONTENT;
            return NO_REQUEST;
        }
        return GET_REQUEST;
    } else if (strncasecmp(text, "Connection:", 11) == 0) {
        text += 11;
        text += strspn(text, " \t");
        if (strcasecmp(text, "keep-alive") == 0)
            m_linger = true;
    } else if (strncasecmp(text, "Content-length:", 15) == 0) {
        text += 15;
        text += strspn(text, " \t");
        m_content_length = atol(text);
    } else if (strncasecmp(text, "Host:", 5) == 0) {
        text += 5;
        text += strspn(text, " \t");
        m_host = text;
    }
    return NO_REQUEST;
}

http_conn::HTTP_CODE http_conn::parse_content(char *text)
{
    if (m_read_idx >= (m_content_length + m_checked_idx)) {
        text[m_content_length] = '\0';
        // text 现在指向完整的 POST body（JSON 字符串）
        // m_string 在原代码中指向这里，我们直接复用 m_url 来保存指针
        return GET_REQUEST;
    }
    return NO_REQUEST;
}

http_conn::HTTP_CODE http_conn::process_read()
{
    LINE_STATUS line_status = LINE_OK;
    HTTP_CODE ret = NO_REQUEST;
    char *text = 0;

    while ((m_check_state == CHECK_STATE_CONTENT && line_status == LINE_OK) ||
           ((line_status = parse_line()) == LINE_OK)) {
        text = get_line();
        m_start_line = m_checked_idx;

        switch (m_check_state) {
        case CHECK_STATE_REQUESTLINE:
            ret = parse_request_line(text);
            if (ret == BAD_REQUEST) return BAD_REQUEST;
            break;
        case CHECK_STATE_HEADER:
            ret = parse_headers(text);
            if (ret == BAD_REQUEST) return BAD_REQUEST;
            else if (ret == GET_REQUEST) return do_request();
            break;
        case CHECK_STATE_CONTENT:
            ret = parse_content(text);
            if (ret == GET_REQUEST) return do_request();
            line_status = LINE_OPEN;
            break;
        default:
            return INTERNAL_ERROR;
        }
    }
    return NO_REQUEST;
}

// ===== do_request —— 推理网关路由（连线 BatchScheduler + MockEngine）=====

http_conn::HTTP_CODE http_conn::do_request()
{
    // 只接受 POST /infer
    if (m_method != POST || strcmp(m_url, "/infer") != 0) {
        return NO_RESOURCE;
    }

    // 提取 POST body
    // HTTP 解析器已把 \r\n 替换为 \0，解析完成后 m_start_line 指向 body 起始位置
    if (m_content_length <= 0) {
        return BAD_REQUEST;
    }
    char *body = m_read_buf + m_start_line;

    // 如果调度器未初始化，回退到错误响应
    if (!scheduler) {
        snprintf(m_response_body, sizeof(m_response_body),
                 "{\"status\":\"error\",\"msg\":\"scheduler not initialized\"}");
        return JSON_RESPONSE;
    }

    // 1. 解析 JSON 请求体
    std::vector<float> input_data = json_get_float_array(body, "input");
    char *model = json_get_string(body, "model");

    if (input_data.empty()) {
        snprintf(m_response_body, sizeof(m_response_body),
                 "{\"status\":\"error\",\"msg\":\"missing or empty 'input' field\"}");
        return JSON_RESPONSE;
    }

    // 2. 创建推理任务
    InferenceTask task;
    task.model_name = model ? model : "default";
    task.input_data = std::move(input_data);
    task.priority = 1;          // 默认中优先级
    task.deadline = time(nullptr) + 30;  // 30 秒超时
    task.client_fd = m_sockfd;

    // 3. 投递到调度器，阻塞等待结果
    scheduler->enqueue(&task);
    task.wait();

    // 4. 构造 JSON 响应
    if (task.timeout) {
        snprintf(m_response_body, sizeof(m_response_body),
                 "{\"status\":\"error\",\"msg\":\"inference timeout\"}");
    } else if (!task.success) {
        snprintf(m_response_body, sizeof(m_response_body),
                 "{\"status\":\"error\",\"msg\":\"%s\"}",
                 task.error_msg.empty() ? "inference failed" : task.error_msg.c_str());
    } else {
        // 拼接输出数组
        std::string out = "[";
        for (size_t i = 0; i < task.output_data.size(); ++i) {
            if (i > 0) out += ",";
            char num[64];
            snprintf(num, sizeof(num), "%f", task.output_data[i]);
            out += num;
        }
        out += "]";

        snprintf(m_response_body, sizeof(m_response_body),
                 "{\"status\":\"ok\",\"model\":\"%s\",\"output\":%s}",
                 task.model_name.c_str(), out.c_str());
    }

    return JSON_RESPONSE;
}

// ===== 响应构造 =====

void http_conn::unmap()
{
    if (m_file_address) {
        munmap(m_file_address, m_file_stat.st_size);
        m_file_address = 0;
    }
}

bool http_conn::write()
{
    if (bytes_to_send == 0) {
        modfd(m_epollfd, m_sockfd, EPOLLIN, m_TRIGMode);
        init();
        return true;
    }

    while (1) {
        int temp = writev(m_sockfd, m_iv, m_iv_count);
        if (temp < 0) {
            if (errno == EAGAIN) {
                modfd(m_epollfd, m_sockfd, EPOLLOUT, m_TRIGMode);
                return true;
            }
            unmap();
            return false;
        }

        bytes_have_send += temp;
        bytes_to_send -= temp;

        if (bytes_have_send >= m_iv[0].iov_len) {
            m_iv[0].iov_len = 0;
            m_iv[1].iov_base = m_file_address + (bytes_have_send - m_write_idx);
            m_iv[1].iov_len = bytes_to_send;
        } else {
            m_iv[0].iov_base = m_write_buf + bytes_have_send;
            m_iv[0].iov_len -= bytes_have_send;
        }

        if (bytes_to_send <= 0) {
            unmap();
            modfd(m_epollfd, m_sockfd, EPOLLIN, m_TRIGMode);
            if (m_linger) {
                init();
                return true;
            } else {
                return false;
            }
        }
    }
}

bool http_conn::add_response(const char *format, ...)
{
    if (m_write_idx >= WRITE_BUFFER_SIZE) return false;

    va_list arg_list;
    va_start(arg_list, format);
    int len = vsnprintf(m_write_buf + m_write_idx,
                        WRITE_BUFFER_SIZE - 1 - m_write_idx,
                        format, arg_list);
    if (len > WRITE_BUFFER_SIZE - 1 - m_write_idx) {
        va_end(arg_list);
        return false;
    }
    m_write_idx += len;
    va_end(arg_list);
    return true;
}

bool http_conn::add_status_line(int status, const char *title)
{
    return add_response("%s %d %s\r\n", "HTTP/1.1", status, title);
}
bool http_conn::add_headers(int content_len)
{
    return add_content_length(content_len) && add_linger() && add_blank_line();
}
bool http_conn::add_content_length(int content_len)
{
    return add_response("Content-Length:%d\r\n", content_len);
}
bool http_conn::add_content_type()
{
    return add_response("Content-Type:%s\r\n", "text/html");
}
bool http_conn::add_linger()
{
    return add_response("Connection:%s\r\n",
                        (m_linger == true) ? "keep-alive" : "close");
}
bool http_conn::add_blank_line()
{
    return add_response("%s", "\r\n");
}
bool http_conn::add_content(const char *content)
{
    return add_response("%s", content);
}

bool http_conn::process_write(HTTP_CODE ret)
{
    switch (ret) {
    case INTERNAL_ERROR:
        add_status_line(500, error_500_title);
        add_headers(strlen(error_500_form));
        if (!add_content(error_500_form)) return false;
        break;
    case NO_RESOURCE:
        add_status_line(404, error_404_title);
        add_headers(strlen(error_404_form));
        if (!add_content(error_404_form)) return false;
        break;
    case BAD_REQUEST:
        add_status_line(400, error_400_title);
        add_headers(strlen(error_400_form));
        if (!add_content(error_400_form)) return false;
        break;
    case FORBIDDEN_REQUEST:
        add_status_line(403, error_403_title);
        add_headers(strlen(error_403_form));
        if (!add_content(error_403_form)) return false;
        break;
    case FILE_REQUEST:
        add_status_line(200, ok_200_title);
        if (m_file_stat.st_size != 0) {
            add_headers(m_file_stat.st_size);
            m_iv[0].iov_base = m_write_buf;
            m_iv[0].iov_len = m_write_idx;
            m_iv[1].iov_base = m_file_address;
            m_iv[1].iov_len = m_file_stat.st_size;
            m_iv_count = 2;
            bytes_to_send = m_write_idx + m_file_stat.st_size;
            return true;
        } else {
            const char *ok_string = "<html><body></body></html>";
            add_headers(strlen(ok_string));
            if (!add_content(ok_string)) return false;
        }
        break;
    case JSON_RESPONSE:
        // m_response_body 已在 do_request() 中填好，这里直接组装 HTTP 响应
        add_status_line(200, ok_200_title);
        add_response("Content-Type: application/json\r\n");
        add_content_length(strlen(m_response_body));
        add_linger();
        add_blank_line();
        if (!add_content(m_response_body)) return false;
        break;
    default:
        return false;
    }

    // 单 iovec 模式（响应全在 m_write_buf 中）
    m_iv[0].iov_base = m_write_buf;
    m_iv[0].iov_len = m_write_idx;
    m_iv_count = 1;
    bytes_to_send = m_write_idx;
    return true;
}

// ===== process —— 请求处理入口 =====

void http_conn::process()
{
    HTTP_CODE read_ret = process_read();

    if (read_ret == NO_REQUEST) {
        modfd(m_epollfd, m_sockfd, EPOLLIN, m_TRIGMode);
        return;
    }

    bool write_ret = process_write(read_ret);
    if (!write_ret)
        close_conn();
    modfd(m_epollfd, m_sockfd, EPOLLOUT, m_TRIGMode);
}
