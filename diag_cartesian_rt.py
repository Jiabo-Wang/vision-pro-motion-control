#!/usr/bin/env python3
"""诊断：RT 笛卡尔模式**到底卡在哪一环**。独立脚本，不依赖本项目任何模块。

已知：`try_cartesian_rt.py` 实测 RT 笛卡尔「收指令但不动」（20mm 指令实到 0.4mm，
且和 5mm 时一样，不随指令变）。这个脚本回答**为什么**。

办法：订阅控制器的实时状态，把一条链上的三个点同时看住 ——

    pos_c   指令末端位姿   → 控制器**收没收到**我发的笛卡尔目标
    q_c     指令关节角     → 控制器**算没算出**逆解
    q_m     实测关节角     → 臂**执行没执行**

三种结局对应三个不同的根因：

    pos_c 不变                  → 指令根本没进控制器的规划器
    pos_c 变了、q_c 不变        → **控制器内部逆解失败**（多半是模型库不支持 AR5）
    q_c 变了、q_m 不变          → 逆解成了但执行被挡（限位/使能/安全）

⚠ Python 绑定的 `RtSupportedFields` 只暴露了 8 个常量，但
  `startReceiveRobotState` 收的是**原始字符串**，所以 "pos_c"/"q_c" 这些
  C++ 头文件里有、Python 没导出的字段，直接传字符串就能订阅。
  （字段名见 xCoreSDK_cpp 的 include/rokae/data_types.h）

    python diag_cartesian_rt.py           # 只发当前位姿，零位移
    python diag_cartesian_rt.py --go      # 再发一个 20mm 偏移（会动臂）
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

SDK = "/home/crp-5070ti-01/yuhang_workspace/xCoreSDK-Python"
# 想看的字段。名字来自 C++ 头 data_types.h，Python 侧没导出也能用字符串订阅。
FIELDS = ["q_m", "q_c", "pos_m", "pos_c"]


def log(m):
    print(m, flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="诊断 RT 笛卡尔卡在哪一环")
    p.add_argument("--ip", default="192.168.2.160")
    p.add_argument("--local-ip", default="192.168.2.222")
    p.add_argument("--go", action="store_true", help="发真实偏移（会动臂）")
    p.add_argument("--move", type=float, default=20.0, help="偏移毫米")
    p.add_argument("--secs", type=float, default=2.0, help="每段持续秒数")
    args, _ = p.parse_known_args()

    sys.path.insert(0, str(Path(SDK) / "Release" / "linux"))
    import xCoreSDK_python as sdk
    import datetime as dt

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

    rt = None
    subscribed = False
    try:
        ec = {}
        log(f"已连接 {robot.robotInfo(ec).type}\n")

        # 逐个字段试订阅，找出这台臂到底支持哪些
        log("【订阅字段】逐个试，看控制器认哪些")
        ok_fields = []
        for f in FIELDS:
            try:
                robot.startReceiveRobotState(dt.timedelta(milliseconds=1), [f])
                robot.stopReceiveRobotState()
                ok_fields.append(f)
                log(f"  ✓ {f}")
            except Exception as exc:                    # noqa: BLE001
                log(f"  ✗ {f}: {str(exc)[:70]}")
        if not ok_fields:
            log("  一个都订阅不了，没法诊断。")
            return 2

        ec = {}
        robot.setOperateMode(sdk.OperateMode.automatic, ec)
        ec = {}
        robot.setPowerState(True, ec)
        ec = {}
        if not str(robot.powerState(ec)).endswith("on"):
            log("\n✗ 没上电，停。")
            return 2
        ec = {}
        robot.setMotionControlMode(sdk.MotionControlMode.RtCommandMode, ec)

        robot.startReceiveRobotState(dt.timedelta(milliseconds=1), ok_fields)
        subscribed = True
        log(f"\n已订阅 {ok_fields}")

        why = {}

        def snap(verbose=False):
            """读一帧状态，返回 {字段: 数组}。读失败时把原因记下来，别吞。"""
            out = {}
            try:
                got = robot.updateRobotState(dt.timedelta(milliseconds=50))
                if verbose:
                    log(f"    updateRobotState 收到 {got} 字节")
                if not got:
                    why["_update"] = "updateRobotState 返回 0（超时没收到数据）"
                    return out
            except Exception as exc:                    # noqa: BLE001
                why["_update"] = f"{type(exc).__name__}: {exc}"
                return out
            for f in ok_fields:
                n = 16 if f.startswith("pos") and not f.endswith("abc_m") else 7
                buf = sdk.PyTypeVectorDouble()
                try:
                    # ⚠ PyTypeVectorDouble 不是可迭代对象，`list(buf)` 会抛
                    # TypeError。取值要用 .content() / .get()。
                    robot.getStateData(f, buf, n)
                    vals = buf.content()
                    out[f] = np.array(list(vals), dtype=float)
                except Exception as exc:                # noqa: BLE001
                    why[f] = f"{type(exc).__name__}: {str(exc)[:80]}"
            return out

        ec = {}
        cp0 = robot.cartPosture(sdk.CoordinateType.flangeInBase, ec)
        rt = robot.getRtMotionController()
        rt.startMove(sdk.RtControllerMode.cartesianPosition)
        log("✓ startMove(cartesianPosition)")

        def rpy_R(r):
            cr, sr, cp_, sp, cy, sy = (np.cos(r[0]), np.sin(r[0]), np.cos(r[1]),
                                       np.sin(r[1]), np.cos(r[2]), np.sin(r[2]))
            return np.array([[cy * cp_, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                             [sy * cp_, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                             [-sp, cp_ * sr, cp_ * cr]])

        def cmd(trans):
            c = sdk.CartesianPosition()
            c.trans = list(trans)
            c.rpy = list(cp0.rpy)
            T = np.eye(4)
            T[:3, :3] = rpy_R(cp0.rpy)
            T[:3, 3] = trans
            c.pos = T.reshape(-1).tolist()
            c.elbow = cp0.elbow
            c.hasElbow = True
            c.confData = list(cp0.confData)
            return c

        base = np.array(cp0.trans)
        segs = [("保持当前位姿", base)]
        if args.go:
            segs.append((f"偏移 {args.move:.0f}mm", base + [args.move / 1000.0, 0, 0]))

        for name, target in segs:
            log(f"\n【{name}】目标 trans = {np.round(target*1000, 2).tolist()} mm")
            first = snap(verbose=True)
            if why:
                for k, v in why.items():
                    log(f"    读取失败 {k}: {v}")
                why.clear()
            t0 = time.time()
            n = 0
            while time.time() - t0 < args.secs:
                rt.sendCommand(cmd(target))
                n += 1
                time.sleep(0.001)
            last = snap()
            log(f"  发了 {n} 帧")
            for f in ok_fields:
                if f not in first or f not in last:
                    log(f"  {f:<7} 读不到")
                    continue
                a, b = first[f], last[f]
                if f.startswith("pos"):
                    da = np.linalg.norm(b[3::4][:3] - a[3::4][:3]) * 1000 \
                        if len(b) == 16 else np.linalg.norm(b[:3] - a[:3]) * 1000
                    cur = b[3::4][:3] * 1000 if len(b) == 16 else b[:3] * 1000
                    log(f"  {f:<7} 现在 {np.round(cur,2).tolist()} mm   本段变化 {da:7.3f} mm")
                else:
                    log(f"  {f:<7} 本段变化 {np.degrees(np.abs(b-a)).max():7.4f}°")

        log("\n" + "=" * 62)
        log("怎么读：")
        log("  pos_c 没跟着目标走      → 指令没进控制器规划器")
        log("  pos_c 走了但 q_c 不动   → **控制器内部逆解失败**（模型库不支持 AR5）")
        log("  q_c 动了但 q_m 不动     → 逆解成了，执行被挡")
        return 0
    finally:
        log("\n收尾 ...")
        for nm, fn in (("停 RT", lambda: rt and rt.stopMove()),
                       ("停订阅", lambda: subscribed and robot.stopReceiveRobotState()),
                       ("回 Nrt", lambda: robot.setMotionControlMode(
                           sdk.MotionControlMode.NrtCommandMode, {})),
                       ("断开", lambda: robot.disconnectFromRobot({}))):
            try:
                fn()
                log(f"  ✓ {nm}")
            except Exception as exc:                    # noqa: BLE001
                log(f"  {nm}: {type(exc).__name__}: {str(exc)[:60]}")


if __name__ == "__main__":
    sys.exit(main())
