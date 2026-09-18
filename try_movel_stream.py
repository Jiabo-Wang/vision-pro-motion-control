#!/usr/bin/env python3
"""实验：**本地算绝对目标 → moveAppend(MoveL) → 控制器做逆解和规划**，能不能撑遥操。

**独立脚本，不依赖本项目任何模块。**

动机：实测 `moveAppend` 只要 **0.03ms**（之前一直以为是 42ms —— 那个数其实是
`checkPath` 的，我把两者混了，导致整条排队路径被提前判死）。它只是往队列里塞，
逆解/规划/执行都在控制器上异步做。

配套接口厂家给全了：
    setDefaultConfOpt(False)   逆解选**离当前轴角最近**的解（正是遥操要的分支判据）
    setMaxCacheSize(n)         控制器侧路点缓冲，注释明说给流式短轨迹用
    MoveLCommand.zone          转弯区，相邻段混合，不会一段一停
    WaypointIndex              查队列执行到第几个点 → 能管队列深度

三级，逐级放开：

    L1  连接 + 配置 + 读状态                      不动
    L2  单条 MoveL 走 --move 毫米再回来            **会动**，测「到底动不动」和延迟
    L3  50Hz 连续流式 append（正弦轨迹）           **会动**，测连续性/队列延迟/跟踪误差

默认只到 L1。L2 要 --go，L3 要 --stream。

    python try_movel_stream.py                    # L1
    python try_movel_stream.py --go               # + L2
    python try_movel_stream.py --go --stream      # + L3
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

SDK = "/home/crp-5070ti-01/yuhang_workspace/xCoreSDK-Python"


def log(m):
    print(m, flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="MoveL 流式下发实验（独立）")
    p.add_argument("--ip", default="192.168.2.160")
    p.add_argument("--local-ip", default="192.168.2.222")
    p.add_argument("--go", action="store_true", help="允许 L2 动臂")
    p.add_argument("--stream", action="store_true", help="允许 L3 流式（需同时 --go）")
    p.add_argument("--move", type=float, default=20.0, help="L2 位移(mm)")
    p.add_argument("--amp", type=float, default=25.0, help="L3 正弦幅度(mm)")
    p.add_argument("--secs", type=float, default=8.0, help="L3 时长")
    p.add_argument("--hz", type=float, default=50.0, help="L3 下发频率")
    p.add_argument("--speed", type=float, default=150.0, help="末端线速度 mm/s")
    p.add_argument("--zone", type=float, default=10.0, help="转弯区 mm")
    p.add_argument("--cache", type=int, default=300, help="控制器缓冲路点数")
    p.add_argument("--max-queue", type=int, default=6,
                   help="队列里超过这么多点就**跳过**这一帧（控延迟的关键）")
    args, _ = p.parse_known_args()
    if max(args.move, args.amp) > 60:
        log("位移上限 60mm，这是探路实验")
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
        except Exception as exc:                       # noqa: BLE001
            robot = None
            log(f"  连接 {k}/5: {exc}")
            if k == 5:
                return 1
            time.sleep(12)

    try:
        ec = {}
        log(f"【L1】已连接 {robot.robotInfo(ec).type}")
        ec = {}
        cp0 = robot.cartPosture(sdk.CoordinateType.flangeInBase, ec)
        base = np.array(cp0.trans)
        log(f"  起点 trans = {np.round(base*1000, 2).tolist()} mm")

        # 排队式运动要 Nrt 模式（RT 模式下 Move* 会被拒，报 ec=262）
        ec = {}
        robot.setMotionControlMode(sdk.MotionControlMode.NrtCommandMode, ec)
        ec = {}
        robot.setOperateMode(sdk.OperateMode.automatic, ec)
        ec = {}
        robot.setPowerState(True, ec)
        ec = {}
        if not str(robot.powerState(ec)).endswith("on"):
            log("  ✗ 没上电，停")
            return 2

        for fn, a, nm in ((robot.setDefaultConfOpt, False, "setDefaultConfOpt(False) 取最近解"),
                          (robot.setMaxCacheSize, args.cache, f"setMaxCacheSize({args.cache})"),
                          (robot.setDefaultSpeed, args.speed, f"setDefaultSpeed({args.speed})"),
                          (robot.setDefaultZone, args.zone, f"setDefaultZone({args.zone})")):
            ec = {}
            try:
                fn(a, ec)
                log(f"  ✓ {nm}  ec={ec.get('ec', 0)}")
            except Exception as exc:                   # noqa: BLE001
                log(f"  ✗ {nm}: {type(exc).__name__}: {str(exc)[:70]}")

        def mk(trans):
            t = sdk.CartesianPosition()
            t.trans = list(trans)
            t.rpy = list(cp0.rpy)
            t.elbow = cp0.elbow
            t.hasElbow = True
            t.confData = list(cp0.confData)
            c = sdk.MoveLCommand(t)
            c.speed = args.speed
            c.zone = args.zone
            return c

        def tcp():
            ec = {}
            return np.array(robot.cartPosture(sdk.CoordinateType.flangeInBase, ec).trans)

        def wait_idle(timeout=8.0):
            """等臂真的停下来。不等就发下一段会报 ec=-20「机器人运动中」。
            判据用**位置连续不变**，比 ReachTarget 更可靠（转弯区下不一定报到位）。"""
            t0 = time.perf_counter()
            last = tcp()
            still = 0
            while time.perf_counter() - t0 < timeout:
                time.sleep(0.05)
                cur = tcp()
                if np.linalg.norm(cur - last) < 2e-5:
                    still += 1
                    if still >= 4:
                        return True
                else:
                    still = 0
                last = cur
            return False

        def wp_index():
            try:
                ec = {}
                info = robot.queryEventInfo(sdk.Event.moveExecution, ec)
                return info.get(sdk.EventInfoKey.MoveExecution.WaypointIndex, -1)
            except Exception:                          # noqa: BLE001
                return -1

        if not args.go:
            log("\n【L2】跳过（要动臂加 --go）")
            return 0

        # ── L2: 单条 MoveL，测「动不动」和端到端延迟 ──────────────────
        d = args.move / 1000.0
        log(f"\n【L2】单条 MoveL 走 {args.move:.0f}mm 再回来 —— **臂会动**")
        for tag, target in (("去", base + [d, 0, 0]), ("回", base)):
            ec = {}
            robot.moveReset(ec)
            ec = {}
            robot.moveAppend(mk(target), sdk.PyString(f"seg_{tag}"), ec)
            t0 = time.perf_counter()
            ec = {}
            robot.moveStart(ec)
            if ec.get("ec", 0):
                log(f"  ✗ moveStart 失败 ec={ec.get('ec')} {ec.get('message')}")
                return 3
            # 先测「多久才动起来」，再等它彻底停稳
            t_move = None
            start_p = tcp()
            while time.perf_counter() - t0 < 3.0:
                if np.linalg.norm(tcp() - start_p) > 0.0005:
                    t_move = time.perf_counter() - t0
                    break
                time.sleep(0.002)
            wait_idle()
            got = tcp()
            log(f"  {tag}: 目标 {np.round((target-base)*1000,2).tolist()} mm   "
                f"实到 {np.round((got-base)*1000,2).tolist()} mm   "
                f"耗时 {time.perf_counter()-t0:.2f}s"
                + (f"   首次动起来 {t_move*1000:.0f}ms" if t_move else ""))
        moved = np.linalg.norm(tcp() - base) * 1000
        log(f"  回到起点误差 {moved:.2f} mm")
        log("  → 实到 ≈ 目标 就说明**控制器会算 AR5 的笛卡尔逆解**，"
            "RT 笛卡尔不动是那条通道的问题")

        # ── L3: 流式 ────────────────────────────────────────────────
        if not args.stream:
            log("\n【L3】跳过（要流式加 --stream）")
            return 0

        log(f"\n【L3】{args.hz:.0f}Hz 连续流式 append，正弦 ±{args.amp:.0f}mm，"
            f"{args.secs:.0f} 秒 —— **臂会持续动**")
        ec = {}
        robot.moveReset(ec)
        ec = {}
        robot.moveStart(ec)
        A = args.amp / 1000.0
        period = 1.0 / args.hz
        sent = skipped = 0
        errs = 0
        lags, qs = [], []
        t_start = time.perf_counter()
        last_wp = -1
        while True:
            t = time.perf_counter() - t_start
            if t > args.secs:
                break
            # 队列太深就跳过这一帧：目标是「最新值」，不是必须执行的轨迹
            wp = wp_index()
            depth = sent - (wp + 1) if wp >= 0 else 0
            qs.append(depth)
            if depth > args.max_queue:
                skipped += 1
            else:
                target = base + [A * np.sin(2 * np.pi * 0.25 * t), 0, 0]
                ec = {}
                try:
                    robot.moveAppend(mk(target), sdk.PyString(f"s{sent}"), ec)
                    if ec.get("ec", 0):
                        errs += 1
                    sent += 1
                except Exception:                      # noqa: BLE001
                    errs += 1
                want = A * np.sin(2 * np.pi * 0.25 * t)
                lags.append(abs((tcp() - base)[0] - want) * 1000)
            time.sleep(period)
        log(f"  下发 {sent} 条，跳过 {skipped} 条（队列深），错误 {errs}")
        if qs:
            log(f"  队列深度 中位 {np.median(qs):.1f}  最大 {np.max(qs):.0f}")
        if lags:
            log(f"  跟踪误差 中位 {np.median(lags):.1f}mm  p95 {np.percentile(lags,95):.1f}mm")
        ec = {}
        robot.moveReset(ec)
        # 回起点
        ec = {}
        robot.moveAppend(mk(base), sdk.PyString("home"), ec)
        ec = {}
        robot.moveStart(ec)
        time.sleep(3)
        log(f"  回起点误差 {np.linalg.norm(tcp()-base)*1000:.2f} mm")
        return 0
    finally:
        log("\n收尾 ...")
        for nm, fn in (("moveReset", lambda: robot.moveReset({})),
                       ("断开", lambda: robot.disconnectFromRobot({}))):
            try:
                fn()
                log(f"  ✓ {nm}")
            except Exception as exc:                   # noqa: BLE001
                log(f"  {nm}: {type(exc).__name__}: {str(exc)[:60]}")
        log("  （没有下电）")


if __name__ == "__main__":
    sys.exit(main())
