#!/usr/bin/env python3
"""DLS(AR5IK) vs 优化式(AR5OptIK) —— 按**遥操的用法**比，不是按「随机位姿求解率」比。

区别很重要：遥操里 IK 是每帧跟一小步（--max-lead 4cm），种子是上一帧的解。
衡量的也不是「解得多准」，而是
    1) 给不给得出解（给不出 = rokae_arm return False = 那一帧臂没有目标）
    2) 帧与帧之间关节跳多大（跳 = 不顺）
    3) 一次多少毫秒（50Hz 只有 20ms，还要分给手）

⚠ DH 表是**合成的**（真表要从控制器 getRobotCfg_DHparam 读，臂没连）。
  关节限位用的是 config.py 里的真值——j5/j6 只有 ±0.8727 那个才是关键变量。
  所以下面的绝对数字别当实机指标看，两个求解器之间的**相对**比较是有效的。
"""
import sys, time
import numpy as np

sys.path.insert(0, "/home/crp-5070ti-01/yuhang_workspace/Project_Atom/遥操/VR设备")
sys.path.insert(0, "/home/crp-5070ti-01/yuhang_workspace/vel_ar5/openvla-energy/experiments/robot/ar5")

from vel_kin import AR5Kinematics          # 按文件路径取，避开同名 PyPI 包
from ar5_ik import AR5IK, AR5OptIK

DH = [(0, 0, 342, 0), (-90, 0, 0, 0), (90, 0, 400, 0), (90, 0, 0, 0),
      (-90, 0, 400, 0), (-90, 0, 0, 0), (90, 0, 126, 0)]
JMIN = (-3.1067, -2.0944, -3.1067, -1.0472, -3.1067, -0.8727, -0.8727)
JMAX = (3.1067, 2.0944, 3.1067, 2.5307, 3.1067, 0.8727, 0.8727)
HOME = np.array([0.006413, 0.330561, -0.017369, 1.926388, -0.831742, 0.066788, -0.143514])
MAX_LEAD = 0.04          # 和 avp_arm_teleop.py 的 --max-lead 一致


def axis_angle_to_matrix(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def trajectory(kin, n=1500, seed=0):
    """一段遥操风格的手部轨迹：平移画个李萨如，姿态绕三轴慢摇。

    幅度故意开到**超出工作空间**——遥操里人手一定会走到臂够不到的地方，
    「够不到的时候怎么办」正是这次要比的东西。
    """
    rng = np.random.default_rng(seed)
    T0 = kin.fk(HOME)
    p0, R0 = T0[:3, 3].copy(), T0[:3, :3].copy()
    out = []
    for i in range(n):
        t = i / 50.0                                   # 50Hz
        p = p0 + np.array([0.26 * np.sin(0.55 * t),
                           0.30 * np.sin(0.41 * t + 1.1),
                           0.22 * np.sin(0.67 * t + 2.3)])
        rv = np.array([0.9 * np.sin(0.33 * t), 0.7 * np.sin(0.47 * t + 0.6),
                       1.1 * np.sin(0.29 * t + 1.9)])
        p += rng.normal(0, 0.0008, 3)                  # 头显抖动
        T = np.eye(4)
        T[:3, :3] = axis_angle_to_matrix(rv) @ R0
        T[:3, 3] = p
        out.append(T)
    return out


def run(solver, traj, kin, name):
    q = HOME.copy()
    qs, gaps, jumps, times, nofix = [], [], [], [], 0
    for T_hand in traj:
        # 遥操的 leash：目标不会离当前 TCP 超过 MAX_LEAD
        T_now = kin.fk(q)
        d = T_hand[:3, 3] - T_now[:3, 3]
        nd = np.linalg.norm(d)
        T = T_hand.copy()
        if nd > MAX_LEAD:
            T[:3, 3] = T_now[:3, 3] + d * (MAX_LEAD / nd)

        t0 = time.perf_counter()
        q_new, info = solver.ik(T, q)
        times.append((time.perf_counter() - t0) * 1000.0)

        if q_new is None:
            nofix += 1                      # 驱动会 return False，这一帧没目标
            continue
        jumps.append(float(np.linalg.norm(q_new - q)))
        q = q_new
        qs.append(q.copy())
        e = kin.pose_error(kin.fk(q), T)
        gaps.append(float(np.linalg.norm(e[:3])))

    n = len(traj)
    qs = np.array(qs)
    margin = np.min(np.minimum(qs - np.array(JMIN), np.array(JMAX) - qs)) if len(qs) else np.nan
    print(f"\n── {name} ──")
    print(f"  给解率            {100*(n-nofix)/n:5.1f}%   （丢帧 {nofix}/{n}，丢帧=臂那一帧没目标）")
    print(f"  跟随残差 中位/p95  {1000*np.median(gaps):5.1f} / {1000*np.percentile(gaps,95):5.1f} mm")
    print(f"  帧间跳变 中位/p95  {np.rad2deg(np.median(jumps)):5.2f} / "
          f"{np.rad2deg(np.percentile(jumps,95)):5.2f} °   ← 「顺不顺」")
    print(f"  最大单帧跳变      {np.rad2deg(np.max(jumps)):5.2f} °")
    print(f"  耗时 中位/p95      {np.median(times):5.2f} / {np.percentile(times,95):5.2f} ms"
          f"   （50Hz 预算 20ms）")
    print(f"  离限位最近处      {np.rad2deg(margin):5.2f} °   （<0 就是驱动会拒的解）")
    return dict(give=100*(n-nofix)/n, jump95=np.rad2deg(np.percentile(jumps,95)),
                ms95=np.percentile(times,95), margin=np.rad2deg(margin))


def grad_check(kin):
    """旋转项梯度用了一阶近似，先量一下它到底偏多少。"""
    opt = AR5OptIK(kin)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(300):
        q = np.clip(HOME + rng.normal(0, 0.4, 7), opt.soft_lo, opt.soft_hi)
        T = kin.fk(np.clip(HOME + rng.normal(0, 0.5, 7), opt.soft_lo, opt.soft_hi))
        c0, g = opt._cost_grad(q, T, HOME)
        gn = np.empty(7)
        for i in range(7):
            h = 1e-6
            qp = q.copy(); qp[i] += h
            qm = q.copy(); qm[i] -= h
            gn[i] = (opt._cost_grad(qp, T, HOME)[0] - opt._cost_grad(qm, T, HOME)[0]) / (2*h)
        rel = np.linalg.norm(g - gn) / max(np.linalg.norm(gn), 1e-9)
        cos = float(g @ gn / (np.linalg.norm(g) * np.linalg.norm(gn) + 1e-30))
        worst = max(worst, rel)
        if cos < 0.99:
            print(f"  ⚠ 下降方向偏了: cos={cos:.4f}")
    print(f"梯度校验（对比数值差分，300 个随机姿态）：最大相对误差 {worst:.3%}")
    print("  一阶近似只影响**步长**不影响方向，L-BFGS-B 的线搜索会自己修，"
          "所以这个量级可以用。")


if __name__ == "__main__":
    kin = AR5Kinematics(DH, num_joints=7, joint_min=JMIN, joint_max=JMAX)
    print(f"HOME 处 TCP = {np.round(kin.fk(HOME)[:3,3], 4).tolist()}  "
          f"（合成 DH，不是实机；相对比较有效）")
    grad_check(kin)

    traj = trajectory(kin)
    print(f"\n轨迹 {len(traj)} 帧，50Hz ≈ {len(traj)/50:.0f} 秒的遥操，"
          f"幅度故意超出工作空间")

    a = run(AR5IK(kin), traj, kin, "DLS 雅可比迭代 (AR5IK，现在的默认)")
    b = run(AR5OptIK(kin), traj, kin, "优化式 (AR5OptIK，chf_ws 路线)")

    print("\n" + "=" * 68)
    print(f"  给解率      {a['give']:.1f}%  →  {b['give']:.1f}%")
    print(f"  帧间跳变p95 {a['jump95']:.2f}° →  {b['jump95']:.2f}°")
    print(f"  耗时 p95    {a['ms95']:.2f}ms →  {b['ms95']:.2f}ms")
    print(f"  限位余量    {a['margin']:.2f}° →  {b['margin']:.2f}°")
