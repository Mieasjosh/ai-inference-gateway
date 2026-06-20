#!/usr/bin/env python3
"""压测对比 —— 单请求顺序 vs 动态批处理 QPS / 延迟分析"""

import urllib.request
import json
import concurrent.futures
import time
import sys
import subprocess
import os
import argparse

EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_gateway")

# ===== 工具函数 =====

def start_server(port, **kwargs):
    """启动 ai_gateway 实例，返回 Popen 对象"""
    cmd = [EXE, "-p", str(port),
           "-t", str(kwargs.get("threads", 8)),
           "-s", str(kwargs.get("engine_ms", 50)),
           "-T", str(kwargs.get("timeout_sec", 30)),
           "-b", str(kwargs.get("batch_ms", 10)),
           "-n", str(kwargs.get("max_batch", 8)),
           "-C", str(kwargs.get("max_concurrent", 2)),
           "-c", "1"]  # 关闭日志减少 IO 干扰
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    return proc


def kill_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def infer(url, payload):
    """单次推理，返回 (latency_ms, success)"""
    data = json.dumps(payload, separators=(',', ':')).encode()
    req = urllib.request.Request(url, data=data,
        headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        elapsed = (time.perf_counter() - start) * 1000
        body = json.loads(resp.read())
        return elapsed, body.get("status") == "ok"
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, False


# ===== 压测模式 =====

def bench_sequential(url, payload, count):
    """顺序发送请求（一次一个，等响应后再发下一个）—— 模拟无批处理的基线"""
    latencies = []
    successes = 0
    t_start = time.perf_counter()

    for _ in range(count):
        lat, ok = infer(url, payload)
        latencies.append(lat)
        if ok:
            successes += 1

    elapsed = time.perf_counter() - t_start
    return {
        "mode": "sequential",
        "total": count,
        "ok": successes,
        "fail": count - successes,
        "elapsed_sec": elapsed,
        "qps": count / elapsed if elapsed > 0 else 0,
        "latencies_ms": latencies,
    }


def bench_concurrent(url, payload, count, max_workers):
    """并发发送请求 —— 触发动态批处理"""
    latencies = []
    successes = 0
    t_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(infer, url, payload) for _ in range(count)]
        for f in concurrent.futures.as_completed(futures):
            lat, ok = f.result()
            latencies.append(lat)
            if ok:
                successes += 1

    elapsed = time.perf_counter() - t_start
    return {
        "mode": f"concurrent (workers={max_workers})",
        "total": count,
        "ok": successes,
        "fail": count - successes,
        "elapsed_sec": elapsed,
        "qps": count / elapsed if elapsed > 0 else 0,
        "latencies_ms": latencies,
    }


# ===== 报告 =====

def percentile(data, p):
    """返回 data 的 p 分位数（p ∈ 0..100）"""
    if not data:
        return 0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    return sorted_data[f]


def report(result):
    """打印单次压测结果"""
    lats = result["latencies_ms"]
    print(f"  模式:        {result['mode']}")
    print(f"  请求数:      {result['total']}")
    print(f"  成功:        {result['ok']}")
    print(f"  失败:        {result['fail']}")
    print(f"  总耗时:      {result['elapsed_sec']:.2f}s")
    print(f"  QPS:         {result['qps']:.1f} req/s")
    if lats:
        print(f"  平均延迟:    {sum(lats)/len(lats):.1f}ms")
        print(f"  P50 延迟:    {percentile(lats, 50):.1f}ms")
        print(f"  P95 延迟:    {percentile(lats, 95):.1f}ms")
        print(f"  P99 延迟:    {percentile(lats, 99):.1f}ms")
    print()


# ===== 主流程 =====

def main():
    parser = argparse.ArgumentParser(
        description="AI 推理网关压测：单请求 vs 批处理")
    parser.add_argument("--port", type=int, default=9988,
                        help="压测服务端口 (default: 9988)")
    parser.add_argument("--engine-ms", type=int, default=100,
                        help="模拟推理延迟 ms (default: 100)")
    parser.add_argument("--batch-ms", type=int, default=20,
                        help="批处理窗口 ms (default: 20)")
    parser.add_argument("--max-batch", type=int, default=8,
                        help="最大批大小 (default: 8)")
    parser.add_argument("--max-concurrent", type=int, default=2,
                        help="最大并发 batch 数 (default: 2)")
    parser.add_argument("--count", type=int, default=40,
                        help="每种模式的请求数 (default: 40)")
    parser.add_argument("--threads", type=int, default=12,
                        help="服务端 worker 线程数 (default: 12)")
    args = parser.parse_args()

    PAYLOAD = {"model": "mock", "input": [1.0, 2.0, 3.0, 4.0]}
    URL = f"http://127.0.0.1:{args.port}/infer"

    print("=" * 60)
    print("AI 推理网关 — 动态批处理压测")
    print("=" * 60)
    print(f"  配置: engine={args.engine_ms}ms, batch_window={args.batch_ms}ms, "
          f"max_batch={args.max_batch}, max_concurrent={args.max_concurrent}")
    print(f"  每种模式 {args.count} 个请求")
    print()

    # 启动服务
    print(f"启动压测服务 (端口 {args.port})...")
    proc = start_server(args.port,
                        threads=args.threads,
                        engine_ms=args.engine_ms,
                        batch_ms=args.batch_ms,
                        max_batch=args.max_batch,
                        max_concurrent=args.max_concurrent)

    results = []
    try:
        # 1. 顺序基线（无批处理优势）
        print("--- 1. 顺序请求（基线，无批处理） ---")
        r_seq = bench_sequential(URL, PAYLOAD, args.count)
        report(r_seq)
        results.append(r_seq)

        # 2. 低并发（模拟日常负载）
        workers_low = max(4, args.max_concurrent * 2)
        print(f"--- 2. 低并发 ({workers_low} workers) ---")
        r_low = bench_concurrent(URL, PAYLOAD, args.count, workers_low)
        report(r_low)
        results.append(r_low)

        # 3. 高并发（模拟峰值负载）
        workers_high = max(8, args.max_concurrent * 4)
        print(f"--- 3. 高并发 ({workers_high} workers) ---")
        r_high = bench_concurrent(URL, PAYLOAD, args.count, workers_high)
        report(r_high)
        results.append(r_high)

        # 汇总对比
        print("=" * 60)
        print("对比汇总")
        print("=" * 60)
        baseline_qps = results[0]["qps"]
        print(f"  {'模式':<25} {'QPS':>8} {'平均延迟':>10} {'加速比':>8}")
        print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8}")
        for r in results:
            speedup = r["qps"] / baseline_qps if baseline_qps > 0 else 0
            avg_lat = sum(r["latencies_ms"]) / len(r["latencies_ms"]) if r["latencies_ms"] else 0
            label = r["mode"].split("(")[0].strip()
            print(f"  {label:<25} {r['qps']:>8.1f} {avg_lat:>10.1f}ms {speedup:>7.1f}x")

        print()
        speedup = results[-1]["qps"] / baseline_qps if baseline_qps > 0 else 0
        print(f"  最大加速比: {speedup:.1f}x")
        print(f"  （高并发 QPS / 顺序 QPS）")

        # 验证
        print()
        if speedup >= 2.0 and all(r["fail"] == 0 for r in results):
            print(f"✓ 批处理加速比 {speedup:.1f}x ≥ 2x，所有请求成功 — PASS")
            return 0
        elif speedup < 2.0:
            print(f"✗ 加速比 {speedup:.1f}x < 2x — 批处理效果不明显，请检查配置")
            return 1
        else:
            print("✗ 有请求失败")
            return 1

    finally:
        kill_server(proc)


if __name__ == "__main__":
    sys.exit(main())
