#!/usr/bin/env python3
"""把控制器里**标定后**的 DH 表读出来存成 JSON。只读，不上电、不运动、不开 RT。

为什么必须读真表：闭式解析逆解是逐项对着 DH 参数推的，参数错一点解就整个错。
仓库里没有存过这张表（`AR5Kinematics.from_robot` 是运行时现读的），
离线写闭式解只能拿编的表凑，没有意义。

标定后的值和名义值在这台臂上能差到 0.4（在末端是毫米级），所以取
`getRobotCfg_DHparam(False)` —— False = 标定后，True = 名义。

    python read_dh.py                 # 读默认 IP，存 ar5_dh.json
    python read_dh.py --out x.json

⚠ 上一个进程刚退就马上连会失败（报「网络异常: 网络连接错误」，和网络无关，
  是会话没释放），实测要等约 40 秒。所以这里带重试。
"""
import argparse
import json
import sys
import time
from pathlib import Path

VEL_AR5 = ("/home/crp-5070ti-01/yuhang_workspace/vel_ar5/openvla-energy/"
           "experiments/robot/ar5")


def main() -> int:
    p = argparse.ArgumentParser(description="读取 AR5 标定后的 DH 表（只读）")
    p.add_argument("--ip", default="192.168.2.160")
    p.add_argument("--local-ip", default="192.168.2.222")
    p.add_argument("--sdk", default="/home/crp-5070ti-01/yuhang_workspace/xCoreSDK-Python")
    p.add_argument("--robot-class", default="xMateErProRobot")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "ar5_dh.json"))
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--retry-wait", type=float, default=15.0)
    args = p.parse_args()

    # 不走 vel_ar5 的 load_sdk：rokae_arm.py 里是 `from ..transforms import ...`，
    # 单独 import 会报「attempted relative import beyond top-level package」。
    # SDK 本身就是个 .so，直接加路径导入即可。
    sdk_dir = Path(args.sdk) / "Release" / "linux"
    if not sdk_dir.is_dir():
        print(f"SDK 目录不存在: {sdk_dir}")
        return 1
    sys.path.insert(0, str(sdk_dir))
    import xCoreSDK_python as sdk  # noqa: E402

    if not hasattr(sdk, args.robot_class):
        avail = sorted(n for n in dir(sdk) if n.endswith("Robot"))
        print(f"SDK 没有 {args.robot_class}，可用的: {avail}")
        return 1
    cls = getattr(sdk, args.robot_class)

    robot = None
    for attempt in range(1, args.retries + 1):
        try:
            robot = cls(args.ip, args.local_ip)
            ec = {}
            robot.connectToRobot(ec)
            if ec.get("ec", 0):
                raise RuntimeError(f"ec={ec.get('ec')}: {ec.get('message')}")
            break
        except Exception as exc:  # noqa: BLE001
            robot = None
            print(f"  第 {attempt}/{args.retries} 次连接失败: {exc}")
            if attempt == args.retries:
                print("\n连不上。检查：")
                print(f"  1) ping {args.ip} 通不通")
                print("  2) 上一个进程是不是刚退 —— 会话没释放，等 40 秒再来")
                print("  3) 本机在 192.168.2.x 网段上有地址吗（ip -br addr）")
                return 1
            print(f"     等 {args.retry_wait:.0f} 秒再试（会话释放要约 40 秒）")
            time.sleep(args.retry_wait)

    try:
        ec = {}
        info = robot.robotInfo(ec)
        print(f"已连接: {info.type}  xCore {info.version}  {info.joint_num} 轴")

        out = {"ip": args.ip, "robot_type": str(info.type),
               "xcore_version": str(info.version), "joint_num": int(info.joint_num),
               "read_at": time.strftime("%Y-%m-%d %H:%M:%S")}

        for tag, nominal in (("calibrated", False), ("nominal", True)):
            ec = {}
            dh = robot.getRobotCfg_DHparam(nominal, ec)
            if ec.get("ec", 0):
                print(f"  {tag}: 读失败 ec={ec.get('ec')} {ec.get('message')}")
                continue
            rows = [list(map(float, r)) for r in dh] if not isinstance(dh[0], float) \
                else [list(map(float, dh[i:i + 4])) for i in range(0, len(dh), 4)]
            out[tag] = rows
            print(f"\n  {tag}  ({len(rows)} 行)   [alpha_deg, a_mm, d_mm, theta_deg]")
            for i, r in enumerate(rows):
                print(f"    行{i}: alpha={r[0]:+9.4f}°  a={r[1]:+9.4f}mm  "
                      f"d={r[2]:+9.4f}mm  theta={r[3]:+9.4f}°")

        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\n已写入 {args.out}")
        if "calibrated" in out and "nominal" in out:
            import itertools
            d = max(abs(a - b) for ra, rb in zip(out["calibrated"], out["nominal"])
                    for a, b in itertools.zip_longest(ra, rb, fillvalue=0.0))
            print(f"标定值与名义值最大差 {d:.4f}（差得越多，越不能用名义值凑）")
        return 0
    finally:
        try:
            ec = {}
            robot.disconnectFromRobot(ec)
            print("已断开（没上电、没运动、没开 RT）")
        except Exception as exc:  # noqa: BLE001
            print(f"断开时: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
