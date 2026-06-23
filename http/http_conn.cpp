#include "http_conn.h"
#include "../scheduler/inference_task.h"
#include "../scheduler/batch_scheduler.h"

#include <string>
#include <cstdlib>
#include <cstring>
#include <exception>

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
int http_conn::task_timeout_sec = 30;

// ===== 简易 JSON 解析（无第三方依赖） =====

// 跳过空白字符，返回第一个非空白字符的位置
static inline char *skip_ws(char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    return p;
}

// 从 JSON 字符串中提取字符串值：搜 "\"key\":" → 跳过空格 → "\"value\""
static std::string json_get_string(char *json, const char *key)
{
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":", key);
    char *start = strstr(json, search);
    if (!start) return "";
    start += strlen(search);
    start = skip_ws(start);          // 跳过 : 后的空格
    if (*start != '"') return "";
    ++start;                         // 跳过开头引号
    char *end = strchr(start, '"');
    if (!end) return "";
    return std::string(start, end - start);
}

// 从 JSON 字符串中提取整数值：搜 "\"key\":" → 跳过空格 → 解析整数
static int json_get_int(char *json, const char *key, int default_val = 0)
{
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":", key);
    char *start = strstr(json, search);
    if (!start) return default_val;
    start += strlen(search);
    start = skip_ws(start);
    if (*start < '0' || *start > '9') return default_val;
    char *end;
    long val = strtol(start, &end, 10);
    if (end == start) return default_val;
    return static_cast<int>(val);
}

// 从 JSON 字符串中提取浮点数组：搜 "\"key\":" → 跳过空格 → "[" → 解析
static std::vector<float> json_get_float_array(char *json, const char *key)
{
    std::vector<float> result;
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":", key);
    char *start = strstr(json, search);
    if (!start) return result;
    start += strlen(search);
    start = skip_ws(start);          // 跳过 : 后的空格
    if (*start != '[') return result;
    ++start;                         // 跳过 [

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
    // 释放旧缓冲区（连接复用时 delete[] nullptr 安全）
    delete[] m_write_buf;
    delete[] m_response_body;

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

    // 动态分配读缓冲区（支持大模型输入，如 MobileNet 150K floats ~1.5MB JSON）
    m_read_buf_size = READ_BUFFER_SIZE;
    m_read_buf = new char[m_read_buf_size];
    memset(m_read_buf, '\0', m_read_buf_size);

    // 动态分配写缓冲区和响应体缓冲区（支持大模型输出，如 MobileNet 1000 floats ~15KB JSON）
    m_write_buf_size = INIT_BUFFER_SIZE;
    m_write_buf = new char[m_write_buf_size];
    memset(m_write_buf, '\0', m_write_buf_size);

    m_response_body_size = INIT_BUFFER_SIZE;
    m_response_body = new char[m_response_body_size];
    memset(m_response_body, '\0', m_response_body_size);

    memset(m_real_file, '\0', FILENAME_LEN);
    m_response_status = 200;
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
    if (m_read_idx >= m_read_buf_size)
        return false;
    int bytes_read = 0;

    if (0 == m_TRIGMode) {  // LT
        bytes_read = recv(m_sockfd, m_read_buf + m_read_idx,
                          m_read_buf_size - m_read_idx, 0);
        m_read_idx += bytes_read;
        if (bytes_read <= 0) return false;
        return true;
    } else {  // ET
        while (true) {
            bytes_read = recv(m_sockfd, m_read_buf + m_read_idx,
                              m_read_buf_size - m_read_idx, 0);
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
        return GET_REQUEST;
    }
    return NO_REQUEST;
}

http_conn::HTTP_CODE http_conn::process_read()
{
    LINE_STATUS line_status = LINE_OK;
    HTTP_CODE ret = NO_REQUEST;
    char *text = 0;

    // 只在非 CONTENT 状态下才调用 parse_line()。
    // CONTENT 状态下 body 可能不含 \r\n（如 JSON），parse_line()
    // 会把 m_checked_idx 扫到 m_read_idx，导致 parse_content() 的
    // 长度校验 (m_content_length + m_checked_idx) 永久无法满足。
    while ((m_check_state == CHECK_STATE_CONTENT && line_status == LINE_OK) ||
           (m_check_state != CHECK_STATE_CONTENT && (line_status = parse_line()) == LINE_OK)) {
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

// ===== do_request —— 推理网关路由 =====

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
        set_response_body("{\"status\":\"error\",\"msg\":\"scheduler not initialized\"}");
        return JSON_RESPONSE;
    }

    // 1. 解析 JSON 请求体
    std::vector<float> input_data = json_get_float_array(body, "input");
    std::string model_name = json_get_string(body, "model");

    if (input_data.empty()) {
        set_response_body("{\"status\":\"error\",\"msg\":\"missing or empty 'input' field\"}");
        return JSON_RESPONSE;
    }

    // 2. 创建推理任务
    InferenceTask task;
    task.model_name = model_name.empty() ? "default" : model_name;
    task.input_data = std::move(input_data);
    task.priority = json_get_int(body, "priority", 1);  // 0=高 1=中 2=低，默认中
    task.deadline = time(nullptr) + task_timeout_sec;
    task.client_fd = m_sockfd;

    // 3. 投递到调度器，阻塞等待结果（带超时）
    // 队列满时返回 503，不阻塞等待
    if (!scheduler->enqueue(&task)) {
        set_response_body("{\"status\":\"error\",\"msg\":\"server overloaded, queue full\"}");
        m_response_status = 503;
        return JSON_RESPONSE;
    }
    task.wait_with_timeout_ms(task_timeout_sec * 1000);

    // 4. 构造 JSON 响应（使用 std::string 自动管理长度）
    if (task.timeout) {
        set_response_body("{\"status\":\"error\",\"msg\":\"inference timeout\"}");
    } else if (!task.success) {
        std::string err_body = "{\"status\":\"error\",\"msg\":\"";
        err_body += task.error_msg.empty() ? "inference failed" : task.error_msg.c_str();
        err_body += "\"}";
        set_response_body(err_body);
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

        std::string body_str = "{\"status\":\"ok\",\"model\":\"";
        body_str += task.model_name;
        body_str += "\",\"output\":";
        body_str += out;
        body_str += "}";
        set_response_body(body_str);
    }

    return JSON_RESPONSE;
}

// ===== set_response_body —— 将响应体写入动态缓冲区（自动扩容） =====

void http_conn::set_response_body(const std::string &body)
{
    size_t needed = body.size() + 1;
    if (needed > (size_t)m_response_body_size) {
        delete[] m_response_body;
        // 扩容到 2 的幂，至少满足 needed
        size_t new_size = (size_t)m_response_body_size;
        if (new_size == 0) new_size = INIT_BUFFER_SIZE;
        while (new_size < needed) new_size *= 2;
        m_response_body_size = (int)new_size;
        m_response_body = new char[m_response_body_size];
    }
    strcpy(m_response_body, body.c_str());
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
    va_list arg_list;

    // 第一次尝试：用现有缓冲区格式化
    va_start(arg_list, format);
    int space = m_write_buf_size - m_write_idx;
    if (space < 1) space = 1;
    int len = vsnprintf(m_write_buf + m_write_idx, space, format, arg_list);
    va_end(arg_list);

    if (len < space) {
        // 缓冲区够大，直接完成
        m_write_idx += len;
        return true;
    }

    // 缓冲区不足，扩容（至少 2 倍，循环直到够大）
    int new_size = m_write_buf_size;
    if (new_size == 0) new_size = INIT_BUFFER_SIZE;
    while (new_size - m_write_idx <= len) new_size *= 2;

    char *new_buf = new char[new_size];
    if (m_write_buf) {
        memcpy(new_buf, m_write_buf, m_write_idx);
        delete[] m_write_buf;
    }
    m_write_buf = new_buf;
    m_write_buf_size = new_size;

    // 重新格式化（扩容后保证成功）
    va_start(arg_list, format);
    vsnprintf(m_write_buf + m_write_idx, m_write_buf_size - m_write_idx, format, arg_list);
    va_end(arg_list);

    m_write_idx += len;
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
        // m_response_status 默认为 200，过载时被设为 503
        if (m_response_status == 503) {
            add_status_line(503, "Service Unavailable");
        } else {
            add_status_line(200, ok_200_title);
        }
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
    try {
        HTTP_CODE read_ret = process_read();

        if (read_ret == NO_REQUEST) {
            modfd(m_epollfd, m_sockfd, EPOLLIN, m_TRIGMode);
            return;
        }

        bool write_ret = process_write(read_ret);
        if (!write_ret)
            close_conn();
        modfd(m_epollfd, m_sockfd, EPOLLOUT, m_TRIGMode);
    } catch (const std::exception &e) {
        LOG_ERROR("http_conn::process exception: %s", e.what());
        close_conn();
    } catch (...) {
        LOG_ERROR("http_conn::process unknown exception");
        close_conn();
    }
}
