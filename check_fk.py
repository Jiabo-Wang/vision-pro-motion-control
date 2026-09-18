#!/usr/bin/env python3
"""把**我算的末端位姿**和**控制器自己报的**对一遍。只读，不动臂、不上电。

这是位姿链上最底层的一环，之前一直没验过：
`AR5Kinematics.fk(q)` 只要和控制器的 `cartPosture()` 对不上，
那么每一个笛卡尔目标从一开始就是错的 —— 逆解再准也是把臂送到错的地方。
所有「反解不对」的现象都得先排除这一条，再谈别的。

三件事：
  1. FK 一致性：同一组关节角，我的 FK vs 控制器的 cartPosture
  2. 逆解自洽：拿控制器报的位姿去逆解，解出来的关节角应当回到原处
  3. TCP 偏置：控制器是按法兰报还是按工具报（`--tcp-z` 有没有被算进去）

    python check_fk.py
    python check_fk.py --samples 5      # 让你手动搬几个姿态，每个都比一次
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VEL_AR5 = ("/home/crp-5070ti-01/yuhang_workspace/vel_ar5/openvla-energy/"
           "experiments/robot/ar5")


def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def ang_between(Ra, Rb):
    R = Ra @ Rb.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main() -> int:
    p = argparse.ArgumentParser(description="核对 FK 与控制器（只读）")
    p.add_argument("--ip", default="192.168.2.160")
    p.add_argument("--local-ip", default="192.168.2.222")
    p.add_argument("--sdk", default="/home/crp-5070ti-01/yuhang_workspace/xCoreSDK-Python")
    p.add_argument("--robot-class", default="xMateErProRobot")
    p.add_argument("--dh", default=str(HERE / "ar5_dh.json"))
    p.add_argument("--samples", type=int, default=1,
                   help=">1 时每次等你手动把臂搬到别的姿态再按回车")
    args, _ = p.parse_known_args()

    sys.path.insert(0, VEL_AR5)
    from vel_kin import AR5Kinematics     # 按文件路径取，避开同名 PyPI 包

    dhf = Path(args.dh)
    if not dhf.exists():
        print(f"缺 {dhf}，先跑 `python read_dh.py`")
        return 1
    dh = json.loads(dhf.read_text(encoding="utf-8"))

    sdk_dir = Path(args.sdk) / "Release" / "linux"
    sys.path.insert(0, str(sdk_dir))
    import xCoreSDK_python as sdk

    cls = getattr(sdk, args.robot_class)
    robot = None
    for attempt in range(1, 6):
        try:
            robot = cls(args.ip, args.local_ip)
            ec = {}
            robot.connectToRobot(ec)
            if ec.get("ec", 0):
                raise RuntimeError(f"ec={ec.get('ec')}: {ec.get('message')}")
            break
        except Exception as exc:                      # noqa: BLE001
            robot = None
            print(f"  连接第 {attempt}/5 次失败: {exc}", flush=True)
            if attempt == 5:
                print("连不上。上个进程刚退的话，会话释放要等约 40 秒。")
                return 1
            time.sleep(12)

    try:
        ec = {}
        info = robot.robotInfo(ec)
        print(f"已连接 {info.type}  {info.joint_num} 轴\n", flush=True)

        jn = (-3.1067, -2.0944, -3.1067, -1.0472, -3.1067, -0.8727, -0.8727)
        jx = (3.1067, 2.0944, 3.1067, 2.5307, 3.1067, 0.8727, 0.8727)
        kins = {t: AR5Kinematics(dh[t], 7, jn, jx) for t in ("calibrated", "nominal")
                if t in dh}

        worst = {t: [0.0, 0.0] for t in kins}
        for s in range(args.samples):
            if s:
                input(f"\n把臂搬到另一个姿态，然后按回车（{s + 1}/{args.samples}）...")
            ec = {}
            q = np.asarray(list(robot.jointPos(ec))[:7], dtype=float)
            ec = {}
            cp = robot.cartPosture(sdk.CoordinateType.flangeInBase, ec)
            p_ctrl = np.asarray(cp.trans, dtype=float)
            R_ctrl = rpy_to_matrix(np.asarray(cp.rpy, dtype=float))

            print(f"\n[样本 {s + 1}] 关节角(deg): {np.round(np.degrees(q), 3).tolist()}")
            print(f"  控制器报的法兰位置: {np.round(p_ctrl * 1000, 3).tolist()} mm")
            for tag, kin in kins.items():
                T = kin.fk(q)
                dp = np.linalg.norm(T[:3, 3] - p_ctrl) * 1000
                da = ang_between(T[:3, :3], R_ctrl)
                worst[tag][0] = max(worst[tag][0], dp)
                worst[tag][1] = max(worst[tag][1], da)
                verdict = "✓ 一致" if dp < 1.0 and da < 0.1 else "✗ **对不上**"
                print(f"  我的 FK({tag:<10}) 位置差 {dp:8.3f} mm   姿态差 {da:7.4f}°   {verdict}")

            # 逆解自洽：拿控制器报的位姿反解，应当回到原来的关节角附近
            kin = kins.get("calibrated") or next(iter(kins.values()))
            T_ctrl = np.eye(4)
            T_ctrl[:3, :3] = R_ctrl
            T_ctrl[:3, 3] = p_ctrl
            sys.path.insert(0, str(HERE))
            from ar5_ik import AR5OptIK
            solver = AR5OptIK(kin, max_step=10.0)     # 不限步，纯看解得准不准
            q_ik, info_ik = solver.ik(T_ctrl, q)
            if q_ik is None:
                print(f"  逆解: **给不出解** ({info_ik.get('reason')})")
            else:
                print(f"  逆解回到原姿态: 关节最大差 "
                      f"{np.degrees(np.abs(q_ik - q).max()):.4f}°   "
                      f"残差 {info_ik['pos_error'] * 1000:.4f}mm "
                      f"{np.degrees(info_ik['rot_error']):.4f}°")

        print("\n" + "=" * 62)
        for tag, (dp, da) in worst.items():
            print(f"  {tag:<12} 最大 位置差 {dp:.3f} mm   姿态差 {da:.4f}°")
        ok = all(v[0] < 1.0 and v[1] < 0.1 for v in worst.values())
        if ok:
            print("  ✓ 运动学和控制器一致 —— 位姿链的底层没问题，问题在上层。")
        else:
            print("  ✗ **运动学和控制器对不上** —— 这是根因，先修这个，")
            print("    上层（映射、滤波、逆解调参）全都白搭。")
            print("    常见原因：DH 表读的是名义值 / 少了法兰或工具偏置 / RPY 约定不同。")
        return 0 if ok else 2
    finally:
        try:
            ec = {}
            robot.disconnectFromRobot(ec)
            print("已断开（没上电、没运动）")
        except Exception as exc:                      # noqa: BLE001
            print(f"断开时: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
