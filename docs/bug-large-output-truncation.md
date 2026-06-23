# Bug: 大模型输出被静默截断（MobileNet 1000 分类不可用）

## 症状

- `test_model.onnx`（4 float 输出）通过 C++ 网关正常返回
- `mobilenetv2-7-clean.onnx`（1000 float 输出）通过 C++ 网关时，客户端收到的 JSON 被截断或连接断开
- `demo_mobilenet.py --server` 模式报 `JSONDecodeError` 或返回远小于 1000 维的 output
- `demo_mobilenet.py`（Python 本地推理）正常，排除模型文件问题
- 错误响应（超时/过载/缺少字段）均正常，仅**成功响应**（含大 output 数组）触发

## 根因

HTTP 响应路径上有两个串联的静态 1KB 缓冲区，对大模型输出不够：

### 瓶颈 1：`m_response_body[1024]`

`do_request()` 中成功响应的 `snprintf` 将 JSON 写入 1024 字节的静态数组：

```cpp
char m_response_body[1024];  // http_conn.h:179

// do_request() 成功路径:
snprintf(m_response_body, sizeof(m_response_body),
         "{\"status\":\"ok\",\"model\":\"%s\",\"output\":%s}",
         task.model_name.c_str(), out.c_str());
```

`snprintf` 在输出超过缓冲区大小时**静默截断**，不报错。MobileNet 的 1000 个 float 序列化后约 15KB，实际只有前面 ~68 个 float 被写入响应。

### 瓶颈 2：`m_write_buf[1024]`

即使 `m_response_body` 能装下完整 body，`process_write()` 在 `m_write_buf[1024]` 中组装完整 HTTP 响应（状态行 + 头部 + body）。HTTP 头部约 100 字节，留给 body 的空间只有 ~920 字节。

```cpp
char m_write_buf[WRITE_BUFFER_SIZE];  // http_conn.h:153, WRITE_BUFFER_SIZE=1024

// add_response() 在缓冲区满时返回 false:
if (len > WRITE_BUFFER_SIZE - 1 - m_write_idx) {
    va_end(arg_list);
    return false;  // → process_write() 返回 false → close_conn()
}
```

`add_response()` 返回 false 后，`process_write()` 返回 false，`process()` 直接调用 `close_conn()`，客户端侧表现为连接断开（`RemoteDisconnected`）。

### 为什么小模型没触发

`test_model.onnx` 输出只有 4 个 float，JSON body 约 60 字节，远小于 1KB 限制。MockEngine 默认 `io_size(4, 4)` 同理。只有输出超过 ~68 个 float 才会触发。

## 修复

`m_response_body` 和 `m_write_buf` 从静态数组改为动态分配（初始 64KB，按需自动扩容），照搬 `m_read_buf` 已有的动态分配模式。

### 文件: `http/http_conn.h`

**成员变量改造：**

```cpp
// 修复前
char m_write_buf[WRITE_BUFFER_SIZE];  // 1024
char m_response_body[1024];

// 修复后
char *m_write_buf;
int m_write_buf_size;
char *m_response_body;
int m_response_body_size;
```

**新增初始容量常量：**

```cpp
static const int INIT_BUFFER_SIZE = 65536;  // 64KB
```

**构造/析构适配：**

```cpp
// 修复前
http_conn() : m_read_buf(nullptr), m_read_buf_size(0) {}
~http_conn() { delete[] m_read_buf; }

// 修复后
http_conn() : m_read_buf(nullptr), m_read_buf_size(0),
              m_write_buf(nullptr), m_write_buf_size(0),
              m_response_body(nullptr), m_response_body_size(0) {}
~http_conn() { delete[] m_read_buf; delete[] m_write_buf; delete[] m_response_body; }
```

### 文件: `http/http_conn.cpp`

**`init()` — 动态分配三个缓冲区，连接复用时先释放旧 buffer：**

```cpp
// 修复前
memset(m_write_buf, '\0', WRITE_BUFFER_SIZE);
memset(m_response_body, '\0', sizeof(m_response_body));

// 修复后
delete[] m_write_buf;    // 复用安全（delete[] nullptr 无操作）
delete[] m_response_body;
m_write_buf_size = INIT_BUFFER_SIZE;
m_write_buf = new char[m_write_buf_size];
memset(m_write_buf, '\0', m_write_buf_size);
m_response_body_size = INIT_BUFFER_SIZE;
m_response_body = new char[m_response_body_size];
memset(m_response_body, '\0', m_response_body_size);
```

**`set_response_body()` — 新增方法，自动扩容：**

```cpp
void http_conn::set_response_body(const std::string &body) {
    size_t needed = body.size() + 1;
    if (needed > (size_t)m_response_body_size) {
        delete[] m_response_body;
        size_t new_size = (size_t)m_response_body_size;
        if (new_size == 0) new_size = INIT_BUFFER_SIZE;
        while (new_size < needed) new_size *= 2;
        m_response_body_size = (int)new_size;
        m_response_body = new char[m_response_body_size];
    }
    strcpy(m_response_body, body.c_str());
}
```

**`do_request()` — 6 处 `snprintf(m_response_body, sizeof(...), ...)` 改为 `set_response_body(...)`：**

```cpp
// 修复前（成功路径）：
snprintf(m_response_body, sizeof(m_response_body),
         "{\"status\":\"ok\",\"model\":\"%s\",\"output\":%s}",
         task.model_name.c_str(), out.c_str());

// 修复后：
std::string body_str = "{\"status\":\"ok\",\"model\":\"";
body_str += task.model_name;
body_str += "\",\"output\":";
body_str += out;
body_str += "}";
set_response_body(body_str);
```

错误/超时/过载响应同样改为字符串直接传入，推理失败含动态错误信息时用 `std::string` 拼接后传入。

**`add_response()` — 从「满则返回 false」改为「满则自动扩容重试」：**

```cpp
// 修复前：
if (m_write_idx >= WRITE_BUFFER_SIZE) return false;
// ... vsnprintf
if (len > WRITE_BUFFER_SIZE - 1 - m_write_idx) {
    va_end(arg_list);
    return false;
}

// 修复后：
int space = m_write_buf_size - m_write_idx;
int len = vsnprintf(m_write_buf + m_write_idx, space, format, arg_list);
if (len < space) {
    m_write_idx += len;
    return true;
}
// 扩容：至少 2 倍，循环直到够大
int new_size = m_write_buf_size;
while (new_size - m_write_idx <= len) new_size *= 2;
char *new_buf = new char[new_size];
memcpy(new_buf, m_write_buf, m_write_idx);
delete[] m_write_buf;
m_write_buf = new_buf;
m_write_buf_size = new_size;
// 重新格式化（保证成功）
vsnprintf(m_write_buf + m_write_idx, m_write_buf_size - m_write_idx, format, arg_list);
m_write_idx += len;
```

### 不变的部分

- `process_write()` 中 `m_iv[0].iov_base = m_write_buf` — 指针赋值，动态 buffer 天然兼容
- `write()` 中 `m_iv[0].iov_base = m_write_buf + bytes_have_send` — 偏移计算，同样兼容
- `strlen(m_response_body)` — `strcpy` 保证 null 终止
- `m_read_buf` 已为动态分配（8MB），输入侧无需修改

## 验证

- **编译**：`make clean && make` 零错误
- **回归测试**：`test_phase3.py` 27/28 PASS（1 个已知 §7 竞态失败，非本次修改导致）
- **MobileNet 端到端**：
  ```bash
  # 启动 ONNX 模式
  LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.19.2/lib \
    ./ai_gateway -p 9993 -t 8 -E onnx -M ./mobilenetv2-7-clean.onnx

  # C++ 网关推理
  python demo_mobilenet.py --server http://127.0.0.1:9993/infer
  ```
  输出：推理耗时 220ms，正确返回 Top-5 分类结果（1000 维输出）
- **小输出不退化**：`test_model.onnx`（4 float）正常返回 `y=2x`
- **错误响应不变**：超时/过载/参数缺失等错误响应格式无变化

## 日期

2026-06-23
