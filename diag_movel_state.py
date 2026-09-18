#!/usr/bin/env python3
"""诊断：`operationState` 到底靠不靠得住，`moveStart` 的失败码到底是什么。

**独立脚本，不依赖本项目任何模块。**

为什么要单独查：MoveL 后端的全部逻辑都压在「臂停了没」这一个判据上，
而实测出现了自相矛盾的现象 ——

    · 缓慢走 6mm 只走到 2.96mm，而 `重启=0`、`已在动=0`（moveStart 一次都没成）
      —— 那臂是**怎么动起来的**？
    · 加了「idle 且有欠账就补 start」之后，`moveStart 失败 28 次`，
      而失败码没记下来，不知道是什么。

三个问题分开问：

    Q1  一段 MoveL 走完之后，位置先稳还是 operationState 先退？差多久？
        （如果状态**根本不退**，整个判据就是错的）
    Q2  队列空着的时候调 moveStart，返回什么码？
        （我新加的补救路径会在队列已空时开火，多半就是这个）
    Q3  一段还在跑的时候 append 新点，会被自动接上执行吗？
        （如果会，那「已在动」就真的良性；如果不会，就必须补 start）

全程位移 ≤ 12mm。

    python diag_movel_state.py            # 只做 Q1/Q2（会动臂，≤12mm）
    python diag_movel_state.py --q3       # 加做 Q3
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
    p = argparse.ArgumentParser(description="operationState / moveStart 失败码诊断")
    p.add_argument("--ip", default="192.168.2.160")
    p.add_argument("--local-ip", default="192.168.2.222")
    p.add_argument("--q3", action="store_true", help="加做 Q3（运动中 append）")
    p.add_argument("--move", type=float, default=6.0, help="每段位移 mm")
    p.add_argument("--speed", type=float, default=250.0)
    p.add_argument("--zone", type=float, default=3.0)
    args, _ = p.parse_known_args()
    if args.move > 20:
        log("位移上限 20mm")
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

    def state():
        ec = {}
        return robot.operationState(ec)

    def start():
        """返回 (ec码, message)。"""
        ec = {}
        try:
            robot.moveStart(ec)
        except Exception as exc:                        # noqa: BLE001
            return f"EXC:{type(exc).__name__}", str(exc)[:60]
        return ec.get("ec", 0), ec.get("message", "")

    try:
        ec = {}
        log(f"已连接 {robot.robotInfo(ec).type}\n")
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

        d = args.move / 1000.0
        log(f"静止态 operationState = {state()}   （下面每段走 {args.move:.0f}mm）")

        # ── Q1: 走完之后，位置和状态谁先稳 ──────────────────────────
        log(f"\n【Q1】一段 MoveL 走完，位置先稳还是状态先退？")
        ec = {}
        robot.moveReset(ec)
        ec = {}
        robot.moveAppend(mk(base + [d, 0, 0]), sdk.PyString("q1"), ec)
        code, msg = start()
        log(f"  moveStart → ec={code} {msg}")
        t0 = time.perf_counter()
        t_state_moving = t_state_left = t_pos_stop = None
        last = tcp()
        still = 0
        trace = []
        while time.perf_counter() - t0 < 6.0:
            t = time.perf_counter() - t0
            st = state()
            cur = tcp()
            moving = (st == sdk.OperationState.moving)
            if moving and t_state_moving is None:
                t_state_moving = t
            if t_state_moving is not None and not moving and t_state_left is None:
                t_state_left = t
            if float(np.linalg.norm(cur - last)) < 2e-5:
                still += 1
                if still >= 3 and t_pos_stop is None and t > 0.2:
                    t_pos_stop = t
            else:
                still = 0
            last = cur
            trace.append((t, str(st).split(".")[-1], (cur - base)[0] * 1000))
            if t_state_left is not None and t_pos_stop is not None and t > 1.0:
                break
            time.sleep(0.005)
        log(f"  状态进 moving : {t_state_moving*1000:.0f} ms"
            if t_state_moving is not None else "  状态**从没进过 moving**")
        log(f"  位置停稳     : {t_pos_stop*1000:.0f} ms"
            if t_pos_stop is not None else "  位置一直没停稳")
        log(f"  状态退 moving : {t_state_left*1000:.0f} ms"
            if t_state_left is not None else "  状态**一直没退出 moving**")
        if t_state_left is not None and t_pos_stop is not None:
            lag = (t_state_left - t_pos_stop) * 1000
            log(f"  → 状态比位置晚 {lag:+.0f} ms"
                + ("（正数=状态更保守，正是我们要的）" if lag >= 0
                   else "  ⚠ **状态先退位置还在动**，判据不安全"))
        seen = []
        for _, s_, _ in trace:
            if not seen or seen[-1] != s_:
                seen.append(s_)
        log(f"  状态序列: {' → '.join(seen)}")
        log(f"  实到 {(tcp()-base)[0]*1000:.2f} mm")

        # ── Q2: 队列空着调 moveStart ────────────────────────────────
        log(f"\n【Q2】队列已空时调 moveStart（我新加的补救路径会撞到这个）")
        log(f"  当前状态 {state()}")
        for i in range(3):
            code, msg = start()
            log(f"  第{i+1}次 → ec={code}  {msg}")
            time.sleep(0.05)
        log(f"  位置变化 {np.linalg.norm(tcp()-(base+[d,0,0]))*1000:.3f} mm（应当为 0）")

        # ── Q3: 运动中 append，会不会被自动接上 ──────────────────────
        if args.q3:
            log(f"\n【Q3】一段还在跑的时候 append 新点，会不会被自动接上执行？")
            ec = {}
            robot.moveReset(ec)
            ec = {}
            robot.moveAppend(mk(base), sdk.PyString("q3a"), ec)
            code, msg = start()
            log(f"  起第一段（回 base）moveStart → ec={code} {msg}")
            time.sleep(0.03)                      # 让它跑起来
            log(f"  此刻状态 {state()}")
            ec = {}
            robot.moveAppend(mk(base + [0, d, 0]), sdk.PyString("q3b"), ec)
            log(f"  运动中 append 第二段 → ec={ec.get('ec',0)} {ec.get('message','')}")
            code, msg = start()
            log(f"  紧接着 moveStart → ec={code} {msg}")
            time.sleep(3.0)
            got = tcp() - base
            log(f"  最终位置 {np.round(got*1000,2).tolist()} mm")
            log("  → Y 到了 %.1fmm 说明第二段**被自动接上了**（「已在动」良性）；"
                % (got[1] * 1000)
                + "Y≈0 说明**搁浅了**，必须补 start")

        log("\n收尾：回起点")
        ec = {}
        robot.moveReset(ec)
        ec = {}
        robot.moveAppend(mk(base), sdk.PyString("home"), ec)
        start()
        time.sleep(3.0)
        log(f"  回位误差 {np.linalg.norm(tcp()-base)*1000:.2f} mm")
        return 0
    finally:
        log("\n断开 ...")
        for nm, fn in (("moveReset", lambda: robot.moveReset({})),
                       ("断开", lambda: robot.disconnectFromRobot({}))):
            try:
                fn()
            except Exception as exc:                    # noqa: BLE001
                log(f"  {nm}: {type(exc).__name__}: {str(exc)[:60]}")
        log("  （没有下电）")


if __name__ == "__main__":
    sys.exit(main())
