"""把 AR5 送回起始姿态（关节空间、排队式 MoveAbsJ）。遥操没跑时用；跑着按 `h`。

为什么不开 RT：`setLoad` 在 RT 接管后会被判「机器人未处于空闲状态」(ec=-28706)，
而且回位根本不需要 RT —— goto_joints 在非 RT 下走排队式 MoveAbsJ，
由控制器自己规划，比 RT 逐步伺服温和。关节空间不用 IK，所以从
「IK 不收敛 / 关节已在软限位外」的状态也出得来（那种局面笛卡尔救不回来）。
"""
import sys
import numpy as np

sys.path.insert(0, "/home/crp-5070ti-01/yuhang_workspace/vel_ar5/openvla-energy")
from experiments.robot.ar5.ar5_env import AR5Env          # noqa: E402
from experiments.robot.ar5.config import AR5Config        # noqa: E402

PAYLOAD_KG, PAYLOAD_COM_Z = 1.0, 0.08


def report(env, tag):
    q = np.asarray(env.arm.get_joint_positions(), dtype=float)
    a = env.config.arm
    mg = getattr(a, "joint_limit_margin", 0.05)
    lo = np.asarray(a.joint_min, float) + mg
    hi = np.asarray(a.joint_max, float) - mg
    m = np.minimum(q - lo, hi - q)
    k = int(np.argmin(m))
    print(f"  [{tag}] " + " ".join(f"j{i}:{np.rad2deg(v):+.1f}°" for i, v in enumerate(q)))
    bad = [f"j{i}" for i in range(len(q)) if m[i] < 0]
    print(f"  [{tag}] 最紧 j{k} 余量 {np.rad2deg(m[k]):+.1f}°"
          + (f"   ⚠ **已在软限位外: {', '.join(bad)}**" if bad else ""))


def main():
    c = AR5Config(mock=False, real_hand_in_mock=False)
    c.use_hand = False
    c.arm.use_realtime = False           # ← 关键：不开 RT
    c.cameras.sources = {}
    env = AR5Env(c)
    print("连接 AR5（只连，不开 RT）...", flush=True)
    env.connect()

    r = env.arm._robot
    if hasattr(r, "setLoad"):
        try:
            ec = {}
            r.setLoad(env.arm.sdk.Load(PAYLOAD_KG, [0.0, 0.0, PAYLOAD_COM_Z],
                                       [0.0, 0.0, 0.0]), ec)
            print(f"  [负载] {PAYLOAD_KG}kg 已设 (ec={ec.get('ec',0)})")
        except Exception as e:                            # noqa: BLE001
            print(f"  [负载] 设不了({type(e).__name__})，沿用控制器里已存的设定")
    else:
        print("  [负载] 非 RT 路径没有 setLoad 入口；控制器保存的是上一次的设定"
              "（只有机器人重启才恢复默认），继续")

    report(env, "回位前")
    print("\n  ⚠ 臂即将移动（排队式 MoveAbsJ）...", end=" ", flush=True)
    try:
        ok = env.home_joints()
        print("完成" if ok is not False else "**未在超时内到位**")
    except Exception as e:                                # noqa: BLE001
        print(f"失败: {type(e).__name__}: {e}")
        env.close()
        return 1
    report(env, "回位后")
    env.close()
    print("\n臂已回到起始姿态。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
