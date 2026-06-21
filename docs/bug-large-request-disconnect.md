# Bug: 大请求 body 导致连接断开 (RemoteDisconnected)

## 症状

- 小请求（< 100KB body）正常返回
- 大请求（> 100KB body，如 MobileNet 150K floats ≈ 0.6MB）客户端报 `RemoteDisconnected`
- 服务端进程未崩溃，但连接被关闭
- 边界大约在 5000-10000 floats（100KB-200KB body）

## 根因

`http_conn::process_read()` 的 while 循环在 body 不完整时，条件中的 `parse_line()` 会扫描 body 内容寻找 `\r\n`，副作用是把 `m_checked_idx` 从正确的 body 起始位置污染到 `m_read_idx`。

### 详细分析

`process_read()` 的主循环：

```cpp
while ((m_check_state == CHECK_STATE_CONTENT && line_status == LINE_OK) ||
       ((line_status = parse_line()) == LINE_OK)) {
```

当请求 body 较大、无法在单次 `recv()` 中读完时（localhost TCP 缓冲区约 100KB），分两次或多次读取：

1. 第一次 `read_once()` 读取 ~100KB（不完整）
2. `process_read()` 解析完 headers，进入 `CHECK_STATE_CONTENT`
3. `parse_content()` 发现 body 不完整，返回 `NO_REQUEST`，`line_status = LINE_OPEN`
4. while 循环条件再求值：第一个子句为 false（line_status 不是 LINE_OK）
5. **第二个子句调用 `parse_line()`**——该函数从 `m_checked_idx` 开始扫描，在 body 数据中寻找 `\r\n`
6. Compact JSON body 不含换行符，`parse_line()` 扫描到 `m_read_idx` 然后返回 `LINE_OPEN`
7. **但 `m_checked_idx` 已被递增到 `m_read_idx` 的位置**（即已接收数据的末尾）

下一次 `process_read()` 被调用时：

```cpp
// parse_content() 判断 body 是否完整:
if (m_read_idx >= (m_content_length + m_checked_idx))
//                                ^^^^^^^^^^^^^^
//  m_checked_idx 现在是 ~98404（上次 parse_line 污染的值）
//  实际应该是 ~180（body 起始位置）
//  所以 need = 2976120 + 98404 = 3074524
//  而总数据量只有 ~2976300，永远无法满足
```

**为什么小请求没触发：** localhost 上单次 `recv()` 能读到 ~100KB。body 小于此量的请求在第一次 `process_read()` 就能收齐所有数据，不经过 multi-read 路径。

### 数据流

```
Client 发送 0.6MB POST body
  │
  ├─ recv #1 → ~100KB (不完整)
  │   └─ parse_content: NO_REQUEST
  │       └─ parse_line() 扫描 body，m_checked_idx 被污染 ↑
  │
  ├─ recv #2 → +~1MB
  │   └─ parse_content: need=3074524, have=1106976 → 永远不满足
  │
  ├─ ... 循环直到超时/客户端断开
  │
  └─ 连接关闭 → RemoteDisconnected
```

## 修复

在 while 条件中增加 `m_check_state != CHECK_STATE_CONTENT` 守卫，CONTENT 状态下不再调用 `parse_line()`：

```cpp
// 修复前
while ((m_check_state == CHECK_STATE_CONTENT && line_status == LINE_OK) ||
       ((line_status = parse_line()) == LINE_OK)) {

// 修复后
while ((m_check_state == CHECK_STATE_CONTENT && line_status == LINE_OK) ||
       (m_check_state != CHECK_STATE_CONTENT && (line_status = parse_line()) == LINE_OK)) {
```

文件: `http/http_conn.cpp`，`process_read()` 函数。

## 验证

- 梯度测试：4 / 1000 / 5000 / 10000 / 20000 / 40000 / 80000 / 120000 / 150528 floats 全部 200 OK
- `test_phase3.py` 完整测试套件：27/28 PASS（§7 优先级时序偶发失败是 WSL 已知问题）
- 最大请求 body ~3MB（150528 floats × ~20 bytes/float）正常处理

## 日期

2026-06-21
