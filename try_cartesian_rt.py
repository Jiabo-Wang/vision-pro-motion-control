#!/usr/bin/env python3
"""独立实验：实时**笛卡尔**模式能不能真的驱动 AR5。**不依赖本项目任何模块。**

背景：vel_ar5 的 `_start_realtime()` 注释里写着

    "RT Cartesian is accepted by the controller and produces no motion at all
     (verified), while RT joint tracks exactly"

所以他们退回了关节模式。但同一份代码另一处又记着：

    "A target built from translation and orientation alone leaves `elbow` at 0
     with `hasElbow` false, the controller cannot resolve the inverse kinematics,
     and it drops the command -- silently"

**「接受但不动」正是后面这条的症状。** 7 轴是冗余的：一个末端位姿对应一整族
关节解，控制器需要**臂角 elbow** + **构型 confData** 才能定下来是哪一个。
不填就解不出来，然后静默丢弃 —— 看起来就像"接受了但不动"。

所以这里重试一次，**把 elbow / confData / hasElbow 都填上**。

分四级，每一级都能单独回答一个问题，前三级**零位移**：

    L1  连接 + 读状态（含 elbow/confData）        不上电、不动
    L2  进入 RT cartesianPosition 模式             上电，但不下发目标
    L3  持续下发**当前位姿**（hold）                上电，理论位移 = 0
    L4  下发一个 --move 毫米的小位移再回来          **会动臂**

默认只跑到 L3。要跑 L4 必须显式加 --go。

    python try_cartesian_rt.py                 # L1~L3，不动
    python try_cartesian_rt.py --go            # 加 L4，动 5mm
    python try_cartesian_rt.py --go --move 10  # 动 10mm

════════════════════════════════════════════════════════════════════════════
结论（2026-09-18 在这台 AR5-5_0.8L-W4C5C11 上实测）：**B 不可行。**

RT 笛卡尔模式**接受指令但不产生运动**，和发送方式无关：

    发送方式                                    指令      实到
    ──────────────────────────────────────────────────────────
    trans/rpy, 200Hz                           5mm      0.00mm
    trans/rpy, 1kHz, 带 elbow+confData         5mm      0.00mm
    pos16(齐次矩阵), 1kHz, 带 elbow+confData    5mm      0.40mm
    trans/rpy + pos16, 1kHz, 带 elbow+confData 20mm     0.40mm   ← 不随指令变

最后一行是关键：指令从 5mm 提到 20mm，实到**纹丝不动**还是 0.40mm。
说明那 0.4mm 是固定沉降量，不是跟踪 —— 响应对指令完全无关。

全程 `startMove(cartesianPosition)` 不报错、`sendCommand` 1887 帧零异常。
就是「收了，然后什么都不做」。

**我原来的假设（缺 elbow/confData 导致冗余解不出来）是错的** —— 填上了照样不动。

→ 逆解只能留在本地做。见 `ar5_poe_ik.py`（旋量法解析解）。
════════════════════════════════════════════════════════════════════════════
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

SDK_PATH = "/home/crp-5070ti-01/yuhang_workspace/xCoreSDK-Python"


def log(m):
    print(m, flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="试 RT 笛卡尔模式（独立实验）")
    p.add_argument("--ip", default="192.168.2.160")
    p.add_argument("--local-ip", default="192.168.2.222")
    p.add_argument("--sdk", default=SDK_PATH)
    p.add_argument("--robot-class", default="xMateErProRobot")
    p.add_argument("--go", action="store_true", help="允许 L4 真的动臂")
    p.add_argument("--move", type=float, default=5.0, help="L4 的位移(毫米)")
    p.add_argument("--hz", type=float, default=1000.0,
                   help="下发频率(Hz)。**SDK 文档要求 1ms 间隔**：「发送间隔过长会判断为通信丢包」。之前那次「接受但不动」可能就有这个因素")
    p.add_argument("--hold", type=float, default=2.0, help="L3 保持几秒")
    p.add_argument("--variant", choices=("trans_rpy", "pos16", "both"),
                   default="both",
                   help="位姿怎么填。trans_rpy=只填 trans/rpy（第一次试的，不动）；"
                        "pos16=填 16 元素齐次矩阵（pos 字段）；both=两个都填")
    args, _ = p.parse_known_args()

    if args.move > 30:
        log(f"--move {args.move}mm 太大了，这是个探路实验，上限 30mm")
        return 1

    sys.path.insert(0, str(Path(args.sdk) / "Release" / "linux"))
    import xCoreSDK_python as sdk

    cls = getattr(sdk, args.robot_class)

    # ── L1: 连接 + 读状态 ───────────────────────────────────────────────
    log("【L1】连接并读状态（不上电、不动）")
    robot = None
    for k in range(1, 6):
        try:
            robot = cls(args.ip, args.local_ip)
            ec = {}
            robot.connectToRobot(ec)
            if ec.get("ec", 0):
                raise RuntimeError(f"ec={ec.get('ec')}: {ec.get('message')}")
            break
        except Exception as exc:                       # noqa: BLE001
            robot = None
            log(f"  连接 {k}/5 失败: {exc}")
            if k == 5:
                log("  连不上。上个进程刚退的话，会话释放实测要等约 40 秒。")
                return 1
            time.sleep(12)

    rt = None
    try:
        ec = {}
        info = robot.robotInfo(ec)
        log(f"  已连接 {info.type}  {info.joint_num} 轴")

        ec = {}
        cp0 = robot.cartPosture(sdk.CoordinateType.flangeInBase, ec)
        ec = {}
        q0 = list(robot.jointPos(ec))[:7]
        log(f"  法兰位姿 trans={np.round(cp0.trans, 4).tolist()} "
            f"rpy={np.round(cp0.rpy, 4).tolist()}")
        log(f"  **elbow(臂角) = {cp0.elbow:.6f}   hasElbow = {cp0.hasElbow}**")
        log(f"  **confData = {list(cp0.confData)}**")
        if not cp0.hasElbow:
            log("  ⚠ hasElbow=False —— 之前那次「接受但不动」多半就是栽在这里")

        # ── L2: 进 RT 笛卡尔模式 ────────────────────────────────────────
        log("\n【L2】进入 RT cartesianPosition 模式（会上电，但不下发目标）")
        ec = {}
        robot.setOperateMode(sdk.OperateMode.automatic, ec)
        ec = {}
        robot.setPowerState(True, ec)
        ec = {}
        ps = robot.powerState(ec)
        log(f"  powerState = {ps}")
        if not str(ps).endswith("on"):
            log("  ✗ 没上电。急停没复位 / 示教器占用。停在这里，不往下走。")
            return 2

        ec = {}
        robot.setMotionControlMode(sdk.MotionControlMode.RtCommandMode, ec)
        if ec.get("ec", 0):
            log(f"  ✗ 进 RtCommandMode 失败 ec={ec.get('ec')} {ec.get('message')}")
            return 2
        # 实时报错只能通过 updateRobotState 拿到；不订阅的话错误会被静默覆盖。
        import datetime as _dt
        try:
            robot.startReceiveRobotState(_dt.timedelta(milliseconds=1),
                                         ["tcpPose_m", "jointPos_m", "elbow_m"])
            log("  ✓ 已订阅实时状态（1ms）—— 这样实时报错才拿得到")
            state_on = True
        except Exception as exc:                       # noqa: BLE001
            log(f"  订阅实时状态失败: {type(exc).__name__}: {exc}")
            state_on = False

        rt = robot.getRtMotionController()
        log(f"  拿到 RT 控制器: {type(rt).__name__}")
        log(f"  sendCommand 重载:\n    "
            + (getattr(rt, 'sendCommand').__doc__ or "(无文档)").replace("\n", "\n    ")[:900])

        ec = {}
        try:
            rt.startMove(sdk.RtControllerMode.cartesianPosition)
            log("  ✓ startMove(cartesianPosition) 没报错")
        except Exception as exc:                       # noqa: BLE001
            log(f"  ✗ startMove(cartesianPosition) 抛异常: {exc}")
            log("    → 这台臂的固件不支持 RT 笛卡尔，B 这条路到此为止。")
            return 3

        # ── L3: 下发当前位姿（hold，零位移）────────────────────────────
        log(f"\n【L3】持续下发**当前位姿** {args.hold:.0f} 秒 —— 理论位移 0")
        log("      这一步验的是「指令收不收」，不是「动不动」。")

        def rpy_to_R(r):
            cr, sr, cp_, sp, cy, sy = (np.cos(r[0]), np.sin(r[0]), np.cos(r[1]),
                                       np.sin(r[1]), np.cos(r[2]), np.sin(r[2]))
            return np.array([
                [cy * cp_, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp_, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp_ * sr, cp_ * cr]])

        def make_cmd(trans, rpy):
            c = sdk.CartesianPosition()
            if args.variant in ("trans_rpy", "both"):
                c.trans = list(trans)
                c.rpy = list(rpy)
            if args.variant in ("pos16", "both"):
                # `pos` 是**行优先齐次变换矩阵**。第一次只填了 trans/rpy，
                # pos 全零 —— 如果实时路径读的是 pos，全零矩阵当然被忽略。
                T = np.eye(4)
                T[:3, :3] = rpy_to_R(rpy)
                T[:3, 3] = trans
                c.pos = T.reshape(-1).tolist()
            # 冗余臂带上臂角和构型（第一轮验过：光有这个还不够）
            c.elbow = cp0.elbow
            c.hasElbow = True
            c.confData = list(cp0.confData)
            return c

        base_t = list(cp0.trans)
        base_r = list(cp0.rpy)
        period = 1.0 / args.hz
        sent = errs = 0
        t0 = time.time()
        while time.time() - t0 < args.hold:
            try:
                rt.sendCommand(make_cmd(base_t, base_r))
                sent += 1
            except Exception as exc:                   # noqa: BLE001
                errs += 1
                if errs == 1:
                    log(f"  sendCommand 抛异常: {type(exc).__name__}: {exc}")
            # 实时错误在这里才浮出来（文档原话：有报错会抛出异常）
            if state_on and sent % 200 == 0:
                try:
                    robot.updateRobotState(_dt.timedelta(milliseconds=1))
                except Exception as exc:               # noqa: BLE001
                    if errs == 0:
                        log(f"  ⚠ updateRobotState 报实时错误: "
                            f"{type(exc).__name__}: {exc}")
                    errs += 1
            time.sleep(period)
        ec = {}
        cp1 = robot.cartPosture(sdk.CoordinateType.flangeInBase, ec)
        drift = np.linalg.norm(np.array(cp1.trans) - np.array(cp0.trans)) * 1000
        log(f"  发了 {sent} 帧，异常 {errs} 次，位移 {drift:.3f} mm")
        if errs:
            log("  ✗ sendCommand 不接受 CartesianPosition —— B 这条路走不通。")
            return 3
        log("  ✓ 指令被接受（hold 阶段本来就不该动）")

        # ── L4: 真的动一下 ────────────────────────────────────────────
        if not args.go:
            log(f"\n【L4】跳过（要动臂请加 --go）。"
                f"\n      加上之后会沿基座 +X 走 {args.move:.0f}mm 再回来。")
            return 0

        d = args.move / 1000.0
        log(f"\n【L4】沿基座 +X 走 {args.move:.0f}mm 再回来 —— **臂会动**")
        for tag, target in (("去", [base_t[0] + d, base_t[1], base_t[2]]),
                            ("回", base_t)):
            t0 = time.time()
            while time.time() - t0 < 2.0:
                rt.sendCommand(make_cmd(target, base_r))
                time.sleep(period)
            ec = {}
            cur = robot.cartPosture(sdk.CoordinateType.flangeInBase, ec)
            moved = (np.array(cur.trans) - np.array(base_t)) * 1000
            want = (np.array(target) - np.array(base_t)) * 1000
            log(f"  {tag}: 指令 {np.round(want,2).tolist()} mm  "
                f"实到 {np.round(moved,2).tolist()} mm")

        ec = {}
        cpN = robot.cartPosture(sdk.CoordinateType.flangeInBase, ec)
        back = np.linalg.norm(np.array(cpN.trans) - np.array(cp0.trans)) * 1000
        log(f"\n  回到起点误差 {back:.2f} mm")
        log("  → 如果「去」那一步实到 ≈ 指令，**B 可行**：控制器能在 1kHz 上自己做逆解。")
        log("    如果实到 ≈ 0，那就复现了 vel_ar5 注释里的「接受但不动」，B 不可行。")
        return 0

    finally:
        log("\n收尾 ...")
        for step, fn in (("停 RT", lambda: rt and rt.stopMove()),
                         ("回 Nrt 模式", lambda: robot.setMotionControlMode(
                             sdk.MotionControlMode.NrtCommandMode, {})),
                         ("断开", lambda: robot.disconnectFromRobot({}))):
            try:
                fn()
                log(f"  ✓ {step}")
            except Exception as exc:                   # noqa: BLE001
                log(f"  {step}: {type(exc).__name__}: {exc}")
        log("  （没有下电，臂带电保持位姿）")


if __name__ == "__main__":
    sys.exit(main())
