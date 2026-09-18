#!/usr/bin/env python3
"""实验：`moveStart` 放到工作线程上，能不能既安全又去掉主循环的 42ms 顿挫。

**独立脚本，不改 movel_stream.py。**

── 结论：不行，别再试了（2026-09-18 实测）────────────────────────────
                update()中位  p95      最大    延迟中位  延迟p95   位移
    同步(现状)      1.14ms  43.20ms   87ms     41ms    92ms  39.6mm
    异步            1.18ms  43.81ms   87ms     42ms    90ms  40.0mm

一点没改善。根因在 L1 里：

    工作线程 26503 次读取   主线程 1 次读取     ← 5 秒里只抢到一次

**xCoreSDK 的 Python 绑定在整个 C++ 调用期间不释放 GIL。** 工作线程一进
SDK 调用就把 GIL 攥死，主线程照样被堵 —— 换个线程发 moveStart 毫无意义。
（反过来也成立：任何后台线程做 SDK 调用都会饿死遥操主循环。）
下面的脚本留着是为了让这个结论可复现，不是待办。

问题：MoveL 后端 250mm/s 实测延迟中位 41ms，但 `update()` 耗时
**p95 42.76ms、最大 87ms** —— 因为 `moveStart` 是同步往返 42.8ms，
而它恰好发生在**运动恢复的瞬间**，正是人最能感知延迟的时刻。
50Hz 只有 20ms 预算，这一下要吞掉 2~4 帧。

想法：主循环只管 append，`moveStart` 扔给工作线程。臂启动还是要 42ms，
但主循环不再被堵，期间的新目标能继续进队列。

**但 xCoreSDK 线不线程安全没有文档。** 所以这个脚本先回答两件事：

    A  安全吗   —— 并发调 moveAppend / moveStart / cartPosture，看会不会
                   报错、丢连接、或者返回垃圾数据
    B  值得吗   —— 对比同步 / 异步两种的 update() 耗时分布和实测延迟

分三级：

    L1  只读并发探测（不动臂）：工作线程狂查 cartPosture，主线程也查，
        对比两边读数是否自洽。**零位移。**
    L2  同步基线：走走停停 15 秒，记 update() 耗时和前导距离
    L3  异步：同上，但 moveStart 走工作线程

默认只到 L1。L2/L3 要 --go。

    python try_async_start.py            # L1，不动臂
    python try_async_start.py --go       # 全部，**臂会动**
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

SDK = "/home/crp-5070ti-01/yuhang_workspace/xCoreSDK-Python"


def log(m):
    print(m, flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="moveStart 异步化：安全性 + 收益")
    p.add_argument("--ip", default="192.168.2.160")
    p.add_argument("--local-ip", default="192.168.2.222")
    p.add_argument("--go", action="store_true", help="允许 L2/L3 动臂")
    p.add_argument("--amp", type=float, default=20.0, help="幅度 mm")
    p.add_argument("--secs", type=float, default=15.0, help="每段时长")
    p.add_argument("--speed", type=float, default=250.0)
    p.add_argument("--zone", type=float, default=3.0)
    p.add_argument("--lead-ms", type=float, default=50.0)
    args, _ = p.parse_known_args()
    if args.amp > 60:
        log("幅度上限 60mm，这是探路实验")
        return 1

    sys.path.insert(0, str(Path(SDK) / "Release" / "linux"))
    import xCoreSDK_python as sdk

    robot = None
    for k in range(1, 6):
        try:
            robot = sdk.xMateErProRobot(args.ip, args.local_ip)
            ec = {}
            robot.connectToRobot(ec)
            if ec.get("ec", 0):
                raise RuntimeError(ec)
            break
        except Exception as exc:                        # noqa: BLE001
            robot = None
            log(f"  连接 {k}/5: {exc}")
            if k == 5:
                return 1
            time.sleep(12)

    def tcp():
        ec = {}
        return np.asarray(robot.cartPosture(
            sdk.CoordinateType.flangeInBase, ec).trans, dtype=float)

    try:
        ec = {}
        log(f"已连接 {robot.robotInfo(ec).type}\n")

        # ── L1: 只读并发，不动臂 ──────────────────────────────────────
        log("【L1】两个线程同时查状态 5 秒 —— **臂不动**，只看 SDK 扛不扛得住并发")
        bad = {"exc": 0, "n": 0, "jump": 0}
        stop = threading.Event()

        def hammer():
            last = None
            while not stop.is_set():
                try:
                    cur = tcp()
                    bad["n"] += 1
                    # 静止的臂读数应当纹丝不动。跳变 = 并发读到了撕裂的数据
                    if last is not None and np.linalg.norm(cur - last) > 1e-4:
                        bad["jump"] += 1
                    last = cur
                except Exception:                       # noqa: BLE001
                    bad["exc"] += 1

        th = threading.Thread(target=hammer, daemon=True)
        th.start()
        main_n = main_exc = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 5.0:
            try:
                tcp()
                main_n += 1
            except Exception:                           # noqa: BLE001
                main_exc += 1
            time.sleep(0.002)
        stop.set()
        th.join(2.0)
        log(f"  工作线程 {bad['n']} 次读取，异常 {bad['exc']}，读数跳变 {bad['jump']}")
        log(f"  主线程   {main_n} 次读取，异常 {main_exc}")
        safe = bad["exc"] == 0 and main_exc == 0 and bad["jump"] == 0
        log("  → " + ("✓ 并发只读没问题" if safe
                      else "✗ **并发不安全**，异步化到此为止"))
        if not safe:
            return 2

        if not args.go:
            log("\n【L2/L3】跳过（要动臂加 --go）")
            return 0

        # ── 配置（和 movel_stream.configure 一致）────────────────────
        for nm, fn, a in (("Nrt", robot.setMotionControlMode,
                           sdk.MotionControlMode.NrtCommandMode),
                          ("automatic", robot.setOperateMode,
                           sdk.OperateMode.automatic),
                          ("上电", robot.setPowerState, True),
                          ("取最近解", robot.setDefaultConfOpt, False),
                          ("缓冲", robot.setMaxCacheSize, 300),
                          ("速度", robot.setDefaultSpeed, args.speed),
                          ("转弯区", robot.setDefaultZone, args.zone)):
            ec = {}
            fn(a, ec)
            if ec.get("ec", 0):
                log(f"  ✗ {nm} ec={ec.get('ec')} {ec.get('message','')}")
                return 3

        ec = {}
        cp0 = robot.cartPosture(sdk.CoordinateType.flangeInBase, ec)
        base = np.asarray(cp0.trans, dtype=float)

        def mk(trans):
            t = sdk.CartesianPosition()
            t.trans = [float(v) for v in trans]
            t.rpy = list(cp0.rpy)
            t.elbow = cp0.elbow
            t.hasElbow = True
            t.confData = list(cp0.confData)
            c = sdk.MoveLCommand(t)
            c.speed = args.speed
            c.zone = args.zone
            return c

        def is_idle():
            try:
                ec = {}
                return robot.operationState(ec) != sdk.OperationState.moving
            except Exception:                           # noqa: BLE001
                return False

        max_lead = args.speed * args.lead_ms / 1e6      # 米
        A = args.amp / 1000.0

        def run(mode):
            """mode='sync' | 'async'。返回统计。"""
            ec = {}
            robot.moveReset(ec)
            last_tgt = base.copy()
            dts, leads, xs = [], [], []
            n_app = n_start = n_busy = n_fail = n_inflight = 0
            inflight = threading.Event()
            starter_exc = []

            def do_start():
                ec = {}
                try:
                    robot.moveStart(ec)
                    code = ec.get("ec", 0)
                except Exception as exc:                # noqa: BLE001
                    starter_exc.append(f"{type(exc).__name__}: {exc}")
                    code = 1
                finally:
                    inflight.clear()
                return code

            last_x = 0.0
            t0 = time.perf_counter()
            while True:
                t = time.perf_counter() - t0
                if t > args.secs:
                    break
                w = time.perf_counter()
                # 走 1.2s 停 0.8s —— 模拟真实遥操，逼出重启路径
                if (t % 2.0) < 1.2:
                    last_x = A * np.sin(2 * np.pi * 0.4 * t)
                tgt = base + [last_x, 0, 0]

                if np.linalg.norm(tgt - last_tgt) >= 0.0005:
                    stopped = is_idle()
                    cur = tcp()
                    lead = float(np.linalg.norm(last_tgt - cur))
                    if not (lead > max_lead and not stopped):
                        ec = {}
                        robot.moveAppend(mk(tgt), sdk.PyString(f"a{n_app}"), ec)
                        if not ec.get("ec", 0):
                            n_app += 1
                            last_tgt = tgt
                            if stopped:
                                if mode == "sync":
                                    code = do_start()
                                    if code == -20:
                                        n_busy += 1
                                    elif code:
                                        n_fail += 1
                                    else:
                                        n_start += 1
                                else:
                                    # 已经有一个在飞就别再发，否则会堆线程
                                    if inflight.is_set():
                                        n_inflight += 1
                                    else:
                                        inflight.set()
                                        threading.Thread(
                                            target=do_start, daemon=True).start()
                                        n_start += 1
                cur = tcp()
                xs.append(cur[0] - base[0])
                leads.append(float(np.linalg.norm(last_tgt - cur)) * 1000)
                dts.append((time.perf_counter() - w) * 1000)
                time.sleep(0.02)

            time.sleep(0.5)
            return {"dts": dts, "leads": leads, "xs": xs, "app": n_app,
                    "start": n_start, "busy": n_busy, "fail": n_fail,
                    "inflight": n_inflight, "exc": starter_exc}

        def home():
            ec = {}
            robot.moveReset(ec)
            ec = {}
            robot.moveAppend(mk(base), sdk.PyString("home"), ec)
            ec = {}
            robot.moveStart(ec)
            time.sleep(2.5)

        out = {}
        for lvl, mode, name in (("L2", "sync", "同步(现状)"),
                                ("L3", "async", "异步(工作线程)")):
            log(f"\n【{lvl}】{name} —— **臂会动** {args.secs:.0f} 秒")
            out[mode] = run(mode)
            home()

        log("\n" + "=" * 70)
        log(f"{'':<16}{'update()中位':>12}{'update()p95':>12}{'最大':>8}"
            f"{'延迟中位':>10}{'延迟p95':>9}{'位移':>8}")
        log("-" * 70)
        for mode, name in (("sync", "同步(现状)"), ("async", "异步")):
            r = out[mode]
            d, le, xs = r["dts"], r["leads"], r["xs"]
            log(f"{name:<14}{np.median(d):>11.2f}ms{np.percentile(d,95):>11.2f}ms"
                f"{max(d):>7.0f}ms{np.median(le)/args.speed*1000:>9.0f}ms"
                f"{np.percentile(le,95)/args.speed*1000:>8.0f}ms"
                f"{(max(xs)-min(xs))*1000:>7.1f}mm")
        log("")
        for mode, name in (("sync", "同步"), ("async", "异步")):
            r = out[mode]
            log(f"  {name}: append {r['app']}  重启 {r['start']}  "
                f"已在动 {r['busy']}  失败 {r['fail']}  在飞跳过 {r['inflight']}")
            if r["exc"]:
                log(f"    ⚠ 工作线程异常 {len(r['exc'])} 次: {r['exc'][0][:80]}")
        log("\n判据：异步要同时满足 —— 工作线程零异常、位移不缩水、"
            "update() p95 掉到 20ms 以内。三条缺一条就别上。")
        return 0
    finally:
        log("\n收尾 ...")
        for nm, fn in (("moveReset", lambda: robot.moveReset({})),
                       ("断开", lambda: robot.disconnectFromRobot({}))):
            try:
                fn()
                log(f"  ✓ {nm}")
            except Exception as exc:                    # noqa: BLE001
                log(f"  {nm}: {type(exc).__name__}: {str(exc)[:60]}")
        log("  （没有下电）")


if __name__ == "__main__":
    sys.exit(main())
