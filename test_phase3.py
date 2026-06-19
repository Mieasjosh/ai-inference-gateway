#!/usr/bin/env python3
"""阶段三验证测试 —— 单请求 / 顺序 / 并发 / 错误处理"""

import urllib.request
import json
import concurrent.futures
import time
import sys

URL = "http://127.0.0.1:9999/infer"
PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")

def infer(payload):
    # 使用 compact 格式避免空格，与服务端手写 JSON 解析器兼容
    data = json.dumps(payload, separators=(',', ':')).encode()
    req = urllib.request.Request(URL, data=data,
        headers={"Content-Type": "application/json"})
    start = time.time()
    resp = urllib.request.urlopen(req, timeout=10)
    elapsed = time.time() - start
    return json.loads(resp.read()), elapsed

# === 1. 单请求 ===
print("=== 1. 单请求测试 ===")
r, t = infer({"model": "mock", "input": [1.0, 2.0, 3.0, 4.0]})
print(f"  响应: {json.dumps(r)}")
print(f"  耗时: {t*1000:.1f}ms")
check("status=ok", r.get("status") == "ok")
check("output=2x", r.get("output") == [2.0, 4.0, 6.0, 8.0],
      f"got {r.get('output')}")

# === 2. 顺序请求 ===
print("\n=== 2. 顺序请求测试 (3个) ===")
for i in range(1, 4):
    r, t = infer({"model": "mock", "input": [float(i)] * 4})
    expected = [float(i * 2)] * 4
    ok = r.get("status") == "ok" and r.get("output") == expected
    check(f"请求{i}: output={r.get('output')}", ok,
          f"expected {expected}, {t*1000:.1f}ms")

# === 3. 并发请求 (5个) ===
print("\n=== 3. 并发请求测试 (5个同时) ===")
payload = {"model": "mock", "input": [1.0, 2.0, 3.0, 4.0]}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
    futures = [ex.submit(infer, payload) for _ in range(5)]
    results = [f.result() for f in futures]
success_count = sum(1 for r, _ in results if r.get("status") == "ok")
check(f"全部成功", success_count == 5, f"{success_count}/5 成功")
for i, (r, t) in enumerate(results):
    print(f"  请求{i+1}: status={r.get('status')}, {t*1000:.1f}ms")

# === 4. 大批量并发 (10个) ===
print("\n=== 4. 大批量并发测试 (10个) ===")
payloads = [{"model": "mock", "input": [float(i)] * 4} for i in range(10)]
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
    futures = [ex.submit(infer, p) for p in payloads]
    results = [f.result() for f in futures]
success_count = sum(1 for r, _ in results if r.get("status") == "ok")
check(f"全部成功", success_count == 10, f"{success_count}/10 成功")
for i, (r, t) in enumerate(results):
    expected = [float(i * 2)] * 4
    ok = r.get("output") == expected
    if not ok:
        print(f"  请求{i}: expected {expected}, got {r.get('output')}")

# === 5. 错误请求 ===
print("\n=== 5. 错误处理测试 ===")
r, _ = infer({"model": "mock"})
check("缺少input返回error", r.get("status") == "error",
      f"got {r}")

try:
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:9999/infer"))
except urllib.error.HTTPError as e:
    check("GET请求返回错误", e.code in (400, 404), f"HTTP {e.code}")

# === 结果 ===
print(f"\n{'='*40}")
print(f"结果: {PASS} PASS, {FAIL} FAIL, {PASS+FAIL} total")
if FAIL == 0:
    print("所有测试通过！")
else:
    print(f"{FAIL} 个测试失败！")
sys.exit(0 if FAIL == 0 else 1)
