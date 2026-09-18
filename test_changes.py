#!/usr/bin/env python3
"""离线自检：握拳耦合规则 + 新逆解对驱动的契约。**不需要任何硬件。**

改完 inspire_hand6.clamp 或 ar5_ik.AR5OptIK 之后先跑这个，全绿再上真机。

    python test_changes.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/crp-5070ti-01/yuhang_workspace/vel_ar5/"
                   "openvla-energy/experiments/robot/ar5")

from inspire_hand6 import INDEX, InspireHand6, index_floor  # noqa: E402

FAILED = []


def ck(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗ **FAIL**'} {name}  {detail}")
    if not cond:
        FAILED.append(name)


# ════════════════════════════════════════════════════════════════════════
def test_hand():
    """食指耦合下限：拦「压下去」，放「张开」，永远不把食指往外推。"""
    print("\n【手】握拳时拇指把食指顶开 —— 就是这条在回归")

    def hand(couple=True):
        # serial_factory 返回 None：只测 clamp，不碰串口
        return InspireHand6(serial_factory=lambda: None, couple_index_thumb=couple)

    # 1) 复现用户报的动作：先卷四指（食指到 0），拇指后到
    for couple in (True, False):
        h = hand(couple)
        for t in range(1000, -1, -100):
            h.clamp([t] * 4 + [1000, 100])          # 四指含食指卷到 0
        curl = [h.clamp([0, 0, 0, 0, tb, 100])[INDEX]
                for tb in range(1000, -1, -100)]    # 拇指随后收拢
        ck(f"握拳 couple={couple}: 收拇指全程食指不动", max(curl) == 0,
           f"食指取值 {sorted(set(curl))}")

    # 2) 捏合仍然要拦得住（这条下限本来就是为捏合量的）
    h = hand()
    got = [h.clamp([1000] * 3 + [v, 500, 100])[INDEX] for v in range(1000, -1, -100)]
    ck("捏合(拇指=500): 食指压下去被拦在下限",
       min(got) == index_floor(500), f"最低 {min(got)}, 下限 {index_floor(500)}")

    # 3) 离开拇指的方向永远安全，必须放行
    h = hand()
    for v in range(1000, -1, -100):
        h.clamp([1000] * 3 + [v, 500, 100])         # 先压到下限
    back = [h.clamp([1000] * 3 + [v, 500, 100])[INDEX] for v in (500, 700, 1000)]
    ck("捏住后张开食指: 放行", back == [500, 700, 1000], f"{back}")

    # 4) 拇指让开后食指要能全闭（否则又回到「单独弯食指没反应」）
    h = hand()
    for v in range(1000, -1, -100):
        h.clamp([1000] * 3 + [v, 500, 100])
    ck("拇指让开(800)后食指能全闭", h.clamp([1000] * 3 + [0, 800, 100])[INDEX] == 0)

    # 5) 第一帧没有历史，不许开局就把食指弹出去
    ck("首帧「食指0/拇指0」不推", hand().clamp([0, 0, 0, 0, 0, 100])[INDEX] == 0)

    # 6) 关掉耦合时退化成纯静态限位
    ck("couple=False 只做静态限位",
       hand(False).clamp([-5, 0, 500, 0, 0, 2000]) == [0, 0, 500, 0, 0, 400])


# ════════════════════════════════════════════════════════════════════════
DH = [(0, 0, 342, 0), (-90, 0, 0, 0), (90, 0, 400, 0), (90, 0, 0, 0),
      (-90, 0, 400, 0), (-90, 0, 0, 0), (90, 0, 126, 0)]
JMIN = (-3.1067, -2.0944, -3.1067, -1.0472, -3.1067, -0.8727, -0.8727)
JMAX = (3.1067, 2.0944, 3.1067, 2.5307, 3.1067, 0.8727, 0.8727)
HOME = np.array([0.006413, 0.330561, -0.017369, 1.926388, -0.831742,
                 0.066788, -0.143514])
MARGIN, JUMP_MAX = 0.05, 0.15       # config.joint_limit_margin / joint_jump_max


def aa2m(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def test_ik():
    """新逆解交出去的每个解，都必须过得了驱动的四道闸。"""
    from vel_kin import AR5Kinematics
    from ar5_ik import AR5OptIK

    print("\n【臂】AR5OptIK 对 rokae_arm 的契约")
    kin = AR5Kinematics(DH, 7, JMIN, JMAX)
    s = AR5OptIK(kin, limit_margin=MARGIN, max_step=0.12)
    lo, hi = np.array(JMIN) + MARGIN, np.array(JMAX) - MARGIN

    # 驱动的调用形态：kinematics.ik(T, seed)，两个位置参数
    T = kin.fk(HOME).copy()
    T[:3, 3] += [0.03, 0.02, 0.01]
    q, info = s.ik(T, HOME)
    ck("ik(T, seed) 返回 (q, info)", q is not None and isinstance(info, dict))
    ck("info 带 'reason'（驱动失败时会打印）", "reason" in info)

    # 梯度必须和数值差分一致，否则 L-BFGS-B 的线搜索会乱走
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(100):
        qq = np.clip(HOME + rng.normal(0, 0.4, 7), lo, hi)
        Tt = kin.fk(np.clip(HOME + rng.normal(0, 0.5, 7), lo, hi))
        _, g = s._cost_grad(qq, Tt, HOME)
        gn = np.array([
            (s._cost_grad(np.where(np.arange(7) == i, qq + 1e-6, qq), Tt, HOME)[0]
             - s._cost_grad(np.where(np.arange(7) == i, qq - 1e-6, qq), Tt, HOME)[0]) / 2e-6
            for i in range(7)])
        worst = max(worst, float(np.linalg.norm(g - gn) / max(np.linalg.norm(gn), 1e-9)))
    ck("解析梯度 == 数值差分", worst < 1e-5, f"最大相对误差 {worst:.2e}")

    # 遥操里真实会出现的目标：两条缰绳之内（平移 4cm、姿态 0.35rad）
    rng = np.random.default_rng(1)
    step_worst, solved, out_lim = 0.0, 0, 0
    for _ in range(400):
        seed = np.clip(HOME + rng.normal(0, 0.3, 7), lo, hi)
        Tn = kin.fk(seed)
        Tt = np.eye(4)
        u = rng.normal(0, 1, 3); u /= np.linalg.norm(u)
        w = rng.normal(0, 1, 3); w /= np.linalg.norm(w)
        Tt[:3, 3] = Tn[:3, 3] + u * 0.04
        Tt[:3, :3] = aa2m(w * 0.35) @ Tn[:3, :3]
        qq, _ = s.ik(Tt, seed)
        if qq is None:
            continue
        solved += 1
        step_worst = max(step_worst, float(np.max(np.abs(qq - seed))))
        if np.any(qq < lo - 1e-9) or np.any(qq > hi + 1e-9):
            out_lim += 1
    ck("测试非空（有解才谈得上守约束）", solved > 300, f"给解 {solved}/400")
    ck("闸3 关节限位: 解全在软限位内", out_lim == 0, f"越限 {out_lim} 个")
    ck(f"闸4 跳变: 单关节步长 ≤ 0.12 < {JUMP_MAX}", step_worst <= 0.12 + 1e-9,
       f"实测最大 {step_worst:.5f} rad")

    # 透传：驱动和脚本还会用 base 上的这些
    for a in ("fk", "jacobian", "pose_error", "joint_min", "joint_max",
              "num_joints", "limit_report", "near_singularity", "manipulability"):
        if not hasattr(s, a):
            ck(f"透传 {a}", False)
    ck("属性透传齐全 + fk 与原对象一致", np.allclose(s.fk(HOME), kin.fk(HOME)))


def test_imports():
    """回归：环境里有个同名 PyPI 包 `kinematics`，会把 vel_ar5 的顶掉。

    实机上报出来是 `ModuleNotFoundError: No module named 'kinematics'`
    （或者更隐蔽的 `cannot import name 'AR5Kinematics'`）——取决于调用方
    有没有恰好把 vel_ar5 插在 sys.path 前面。按文件路径加载才是确定的。
    """
    print("\n【导入】同名 PyPI 包不许把 vel_ar5 的运动学顶掉")
    import importlib
    from vel_kin import load_vel_module
    src = load_vel_module("kinematics").__file__
    ck("拿到的是 vel_ar5 的 kinematics.py", "vel_ar5" in src, src)
    # 只做提示，不做断言：`import kinematics` 拿到谁本来就随 sys.path 变，
    # 那正是这里要消除的不确定性 —— 拿它当断言等于把不确定性写进测试。
    try:
        import kinematics as amb
        where = "site-packages（同名 PyPI 包）" if "site-packages" in (amb.__file__ or "") \
                else "vel_ar5"
        print(f"    （参考）裸 `import kinematics` 此刻解析到: {where}")
    except ImportError:
        print("    （参考）裸 `import kinematics` 此刻解析不到任何东西")
    # 每个求解器模块都必须能独立导入，不靠调用方铺路
    for mod in ("ar5_ik", "ar5_srs_ik", "ar5_poe_ik", "paden_kahan", "vel_kin"):
        try:
            importlib.import_module(mod)
            ok = True
        except Exception as exc:                       # noqa: BLE001
            ok = False
            print(f"    {mod}: {type(exc).__name__}: {exc}")
        ck(f"{mod} 可独立导入", ok)


def test_stuck():
    """实机回归：腕部姿态够不到时，臂被快速推进限位然后卡死。

    第一版 AR5OptIK 就是栽在这里 —— 6 帧(120ms)顶死 j5/j6，而且因为它
    永不返回 None，avp_arm_teleop 的两张自救网（打滑重锚 / 限位自救）
    **同时失效**，没有任何人吭声。用户报的「IK 直接快速超限卡住」。
    """
    from vel_kin import AR5Kinematics
    from ar5_ik import AR5OptIK

    print("\n【回归】腕部够不到时不许顶死限位，且必须报得出「跟不上」")
    kin = AR5Kinematics(DH, 7, JMIN, JMAX)
    drv_lo, drv_hi = np.array(JMIN) + MARGIN, np.array(JMAX) - MARGIN
    ML, MRL = 0.04, 0.35        # 遥操的两条缰绳

    def play(T_hand, frames=200):
        """按遥操真实管路跑：先上缰绳，再进 IK。"""
        s = AR5OptIK(kin, limit_margin=MARGIN)
        q = HOME.copy()
        for _ in range(frames):
            Tn = kin.fk(q)
            T = T_hand.copy()
            d = T_hand[:3, 3] - Tn[:3, 3]
            nd = float(np.linalg.norm(d))
            if nd > ML:
                T[:3, 3] = Tn[:3, 3] + d * (ML / nd)
            er = kin.pose_error(Tn, T_hand)[3:]
            nr = float(np.linalg.norm(er))
            if nr > MRL:
                T[:3, :3] = aa2m(er * (MRL / nr)) @ Tn[:3, :3]
            qq, _ = s.ik(T, q)
            if qq is not None:
                q = qq
        return q, s

    # 场景：人手把腕拧到 92°（j5/j6 行程只有 ±50°），然后保持不动
    T_bad = kin.fk(HOME).copy()
    T_bad[:3, :3] = aa2m(np.array([0.0, 0.0, 1.6])) @ T_bad[:3, :3]
    q, s = play(T_bad)
    pinned = [f"j{i}" for i in range(7)
              if q[i] <= drv_lo[i] + 1e-6 or q[i] >= drv_hi[i] - 1e-6]
    margin = float(np.rad2deg(np.min(np.minimum(q - np.array(JMIN),
                                                np.array(JMAX) - q))))
    ck("够不到时不把关节顶死在限位上", not pinned, f"顶死的: {pinned or '无'}")
    ck("留出 ≥5° 余量（不是贴着驱动带子边）", margin >= 5.0, f"最紧 {margin:.2f}°")
    ck("报得出「跟不上」，上层才能重锚", s.stats["saturated"] > 8,
       f"saturated={s.stats['saturated']} (需 > slip_after=8)")

    # 反面：够得着的目标不许误报，否则会无谓地反复重锚
    T_ok = kin.fk(HOME).copy()
    T_ok[:3, 3] += [0.02, 0.01, 0.0]
    _, s2 = play(T_ok, frames=100)
    ck("够得着时不误报「跟不上」", s2.stats["saturated"] == 0,
       f"saturated={s2.stats['saturated']}")


if __name__ == "__main__":
    test_hand()
    test_imports()
    test_ik()
    test_stuck()
    print("\n" + "=" * 60)
    if FAILED:
        print(f"{len(FAILED)} 项失败：" + "、".join(FAILED))
        sys.exit(1)
    print("全部通过。可以上真机了 —— 顺序见 README/回答里的分阶段清单。")
