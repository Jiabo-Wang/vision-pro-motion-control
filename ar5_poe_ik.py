#!/usr/bin/env python3
"""AR5 的**旋量法**解析逆解：PoE 建模 + Paden-Kahan 子问题分解。

和 `ar5_srs_ik.py`（同样是闭式，但按 DH 硬推 ZYZ）的区别：这里所有几何量
都是**基座系里的螺旋轴**，不依赖任何 DH 约定，分支从哪来、有几个是看得见的
——每个子问题自己就带着解的个数。

正运动学（PoE，实测与 DH 等价到 4.4e-16）：

    T(θ) = e^{[S₀]θ₀} e^{[S₁]θ₁} ⋯ e^{[S₆]θ₆} · M

零位螺旋轴（从控制器标定 DH 表算出来的，`read_dh.py` → `ar5_dh.json`）：

    S₀ w=(0,0,1) r=(0,0,0)        ┐
    S₁ w=(0,1,0) r=(0,0,.1745)    ├ 三轴共点于肩心 p_s=(0,0,.1745)
    S₂ w=(0,0,1) r=(0,0,.1745)    ┘
    S₃ w=(0,1,0) r=(.0105,0,.5065)  肘（10.5mm 偏置）
    S₄ w=(0,0,1) r=(0,0,.5065)    ┐
    S₅ w=(0,1,0) r=(0,0,.7845)    ├ 三轴共点于腕心 p_w=(0,0,.7845)
    S₆ w=(1,0,0) r=(0,0,.7845)    ┘

分解思路（每一步都落到一个子问题上）：

  1. **腕心** p_w 不受 S₄S₅S₆ 影响（三轴都穿过它），所以先把它从目标位姿里剥出来。
  2. **肘角 θ₃ —— 子问题 3**：S₀S₁S₂ 都穿过 p_s，转它们不改变 |p_w−p_s|。
     所以 θ₃ 由「绕 S₃ 把 p_w0 转到离 p_s 恰好 δ」唯一决定。**肘上/肘下 2 支**。
  3. **肩 θ₀θ₁θ₂ —— 子问题 2 + 子问题 1**：三个转动把一个点送到指定位置，
     只需 2 个自由度，多出来的那 1 个就是**臂角 ψ**（冗余）。
     固定 ψ 之后，把 S₂ 的方向向量当作待转的点丢给子问题 2 得 (θ₀,θ₁)，
     再用子问题 1 补上 θ₂。**2 支**。
  4. **腕 θ₄θ₅θ₆ —— 同样 子问题 2 + 1**。**2 支**。

  分支上限 2×2×2 = 8。但这台臂 j5/j6 只有 ±50°，腕翻要 j6±180°，
  **恒不可达**（2 万位形实测 0.00%），所以实际拿得到的是 2~4 支。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from paden_kahan import rot, subproblem1, subproblem2, subproblem3


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class AR5PoeIK:
    """旋量法解析逆解。分支全枚举，再按「离上一帧最近」挑。"""

    def __init__(self, dh_json=None, joint_min=None, joint_max=None, *,
                 limit_margin: float = 0.05, limit_backoff: float = 0.03,
                 psi_samples: int = 36, psi_window: int = 3,
                 psi_span: float = 0.40, max_step: float = 0.0148, psi_refine: int = 0,
                 calibrated=None, refine_iters: int = 3, fallback=None, lock=None):
        # 走 vel_kin 按文件路径取，不用 `from kinematics import ...`：
        # 环境里有个同名 PyPI 包会把它顶掉，能不能拿对取决于调用方的 sys.path。
        from vel_kin import AR5Kinematics

        p = Path(dh_json or (Path(__file__).resolve().parent / "ar5_dh.json"))
        dh = json.loads(p.read_text(encoding="utf-8"))
        jn = tuple(joint_min) if joint_min is not None else \
            (-3.1067, -2.0944, -3.1067, -1.0472, -3.1067, -0.8727, -0.8727)
        jx = tuple(joint_max) if joint_max is not None else \
            (3.1067, 2.0944, 3.1067, 2.5307, 3.1067, 0.8727, 0.8727)
        # 螺旋轴从**名义**表提取（严格共点，闭式解才精确成立）；
        # 解完再用**标定**表做几步牛顿，把 0.4 的建模差补回来。
        self._nom = AR5Kinematics(dh["nominal"], 7, jn, jx)
        self.cal = calibrated if calibrated is not None else \
            AR5Kinematics(dh["calibrated"], 7, jn, jx)

        T0, axes, origins = self._nom._chain(np.zeros(7))
        self.w = [np.asarray(a, dtype=float) for a in axes]
        self.r = [np.asarray(o, dtype=float) for o in origins]
        self.M = self._nom.fk(np.zeros(7))
        self.p_s = self.r[1].copy()          # 肩心
        self.p_w0 = self.r[5].copy()         # 零位腕心
        # 腕心在**法兰系**下的固定坐标：把目标位姿往回退就得到腕心
        self.p_w_in_flange = np.linalg.inv(self.M) @ np.append(self.p_w0, 1.0)

        m = limit_margin + limit_backoff
        self.lo = np.asarray(jn, dtype=float) + m
        self.hi = np.asarray(jx, dtype=float) - m
        self.psi_samples, self.psi_window = int(psi_samples), int(psi_window)
        self.psi_span, self.max_step = float(psi_span), float(max_step)
        self.refine_iters = int(refine_iters)
        self.psi_refine = int(psi_refine)
        # 闭式解**只会给精确解或者不给** —— 目标出了工作空间就返回「无分支」，
        # 那一帧被丢掉（激进轨迹实测丢 118/1000）。数值解相反：永远给够得到的
        # 里面最接近的。两者各取所长：能闭式就闭式（精确、快），够不到交给数值解打滑。
        self.fallback = fallback
        # 锁轴：闭式解的关节是分解出来的，没法像 box 约束那样直接钉死。
        # 这里只能**筛**：把不满足锁定值的分支丢掉。ψ 扫过去总会有满足的，
        # 实在没有就退数值解（数值解的 box 约束能真正锁死）。
        self.lock = dict(lock or {})
        self.lock_tol = 1e-3
        self.stats = {"calls": 0, "ok": 0, "no_branch": 0, "branches": 0,
                      "fallback": 0, "lock_filtered": 0}

    # ── 正解（PoE） ─────────────────────────────────────────────────────────
    def fk(self, q):
        T = np.eye(4)
        for i in range(7):
            R = rot(self.w[i], q[i])
            A = np.eye(4)
            A[:3, :3] = R
            A[:3, 3] = self.r[i] - R @ self.r[i]
            T = T @ A
        return T @ self.M

    # ── 三段分解 ────────────────────────────────────────────────────────────
    def _shoulder_angles(self, R_sh) -> List[Tuple[float, float, float]]:
        """从肩部总转动里拆出 (θ₀,θ₁,θ₂)。子问题 2 定前两个，子问题 1 补第三个。

        技巧：S₂ 的方向 w₂ 在 e^{[w₂]θ₂} 下不动，所以 R_sh·w₂ 只由 θ₀θ₁ 决定 ——
        「把 w₂ 转到 R_sh·w₂」正是子问题 2。
        """
        out = []
        tgt = R_sh @ self.w[2]
        for (t0, t1) in subproblem2(self.w[0], self.w[1], np.zeros(3),
                                    self.w[2], tgt):
            R01 = rot(self.w[0], t0) @ rot(self.w[1], t1)
            R2 = R01.T @ R_sh                      # 还剩绕 w₂ 的一转
            # 拿一个不在 w₂ 上的向量把 θ₂ 量出来
            probe = self.w[1] if abs(self.w[1] @ self.w[2]) < 0.9 else self.w[0]
            t2 = subproblem1(self.w[2], np.zeros(3), probe, R2 @ probe)
            if t2:
                out.append((t0, t1, t2[0]))
        return out

    def _wrist_angles(self, R_w) -> List[Tuple[float, float, float]]:
        """腕部同理：S₆ 方向在 e^{[w₆]θ₆} 下不动。"""
        out = []
        tgt = R_w @ self.w[6]
        for (t4, t5) in subproblem2(self.w[4], self.w[5], np.zeros(3),
                                    self.w[6], tgt):
            R45 = rot(self.w[4], t4) @ rot(self.w[5], t5)
            R6 = R45.T @ R_w
            probe = self.w[5] if abs(self.w[5] @ self.w[6]) < 0.9 else self.w[4]
            t6 = subproblem1(self.w[6], np.zeros(3), probe, R6 @ probe)
            if t6:
                out.append((t4, t5, t6[0]))
        return out

    def branches(self, T_d, psi_centre=None, psi_span=None) -> List[np.ndarray]:
        T_d = np.asarray(T_d, dtype=float)
        R_d = T_d[:3, :3]
        p_w_d = (T_d @ self.p_w_in_flange)[:3]     # 目标腕心

        delta = float(np.linalg.norm(p_w_d - self.p_s))
        sols: List[np.ndarray] = []

        # ── 子问题 3：肘角 ───────────────────────────────────────────────
        for t3 in subproblem3(self.w[3], self.r[3], self.p_w0, self.p_s, delta):
            if not (self.lo[3] <= _wrap(t3) <= self.hi[3]):
                continue
            R3 = rot(self.w[3], t3)
            p1 = self.r[3] + R3 @ (self.p_w0 - self.r[3])     # 转完肘之后的腕心
            u, v = p1 - self.p_s, p_w_d - self.p_s
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu < 1e-9 or nv < 1e-9:
                continue
            uh, vh = u / nu, v / nv
            # 任取一个把 uh 转到 vh 的旋转当基准，剩下的自由度就是绕 vh 转 ψ
            c = float(np.clip(uh @ vh, -1, 1))
            if c > 1 - 1e-12:
                R_ref = np.eye(3)
            elif c < -1 + 1e-12:
                a = np.array([1.0, 0, 0]) if abs(uh[0]) < 0.9 else np.array([0, 1.0, 0])
                R_ref = rot(np.cross(uh, a) / np.linalg.norm(np.cross(uh, a)), np.pi)
            else:
                k = np.cross(uh, vh)
                R_ref = rot(k / np.linalg.norm(k), float(np.arccos(c)))

            if psi_centre is None:
                psis = np.linspace(-np.pi, np.pi, self.psi_samples, endpoint=False)
            else:
                sp = self.psi_span if psi_span is None else psi_span
                psis = psi_centre + np.linspace(-sp, sp, self.psi_window)

            for psi in psis:
                R_sh = rot(vh, psi) @ R_ref            # 肩部总转动（含臂角）
                for (t0, t1, t2) in self._shoulder_angles(R_sh):
                    # 腕部还剩多少：把前四个关节的转动剥掉
                    R_arm = (rot(self.w[0], t0) @ rot(self.w[1], t1)
                             @ rot(self.w[2], t2) @ R3)
                    R_w = R_arm.T @ R_d @ self.M[:3, :3].T
                    for (t4, t5, t6) in self._wrist_angles(R_w):
                        q = _wrap(np.array([t0, t1, t2, t3, t4, t5, t6]))
                        if not (np.all(q >= self.lo) and np.all(q <= self.hi)):
                            continue
                        if self.lock and any(abs(q[j] - v) > self.lock_tol
                                             for j, v in self.lock.items()):
                            self.stats["lock_filtered"] += 1
                            continue
                        sols.append(q)
        return sols

    def arm_angle(self, q) -> float:
        """反算一个关节解对应的臂角 ψ。用来给窄窗定中心、也用来自检。"""
        q = np.asarray(q, dtype=float)
        R3 = rot(self.w[3], q[3])
        p1 = self.r[3] + R3 @ (self.p_w0 - self.r[3])
        R_sh = (rot(self.w[0], q[0]) @ rot(self.w[1], q[1]) @ rot(self.w[2], q[2]))
        p_w_d = self.p_s + R_sh @ (p1 - self.p_s)
        u, v = p1 - self.p_s, p_w_d - self.p_s
        uh, vh = u / np.linalg.norm(u), v / np.linalg.norm(v)
        c = float(np.clip(uh @ vh, -1, 1))
        if c > 1 - 1e-12:
            R_ref = np.eye(3)
        elif c < -1 + 1e-12:
            a = np.array([1.0, 0, 0]) if abs(uh[0]) < 0.9 else np.array([0, 1.0, 0])
            R_ref = rot(np.cross(uh, a) / np.linalg.norm(np.cross(uh, a)), np.pi)
        else:
            k = np.cross(uh, vh)
            R_ref = rot(k / np.linalg.norm(k), float(np.arccos(c)))
        M = R_sh @ R_ref.T
        ax = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]) / 2.0
        return float(np.arctan2(ax @ vh, (np.trace(M) - 1.0) / 2.0))

    # ── 驱动接口 ────────────────────────────────────────────────────────────
    def ik(self, T_target, q_seed, **_ignored) -> Tuple[Optional[np.ndarray], dict]:
        self.stats["calls"] += 1
        q_last = np.asarray(q_seed, dtype=float).reshape(7)
        # 窄窗中心取**种子自己的臂角**（无状态，永远对）。ψ 是连续量，
        # 50Hz 下一帧动不了多少，所以窄窗既提速又天然满足「离上一帧最近」。
        try:
            centre = self.arm_angle(q_last)
        except Exception:                      # noqa: BLE001
            centre = None
        cand = self.branches(T_target, psi_centre=centre) if centre is not None else []
        if not cand:
            cand = self.branches(T_target)     # 窄窗空了才全周扫
        self.stats["branches"] += len(cand)
        info = {"branches": len(cand), "reason": ""}
        if not cand:
            self.stats["no_branch"] += 1
            if self.fallback is not None:
                self.stats["fallback"] += 1
                q, fi = self.fallback.ik(T_target, q_last)
                fi = dict(fi)
                fi["reason"] = "闭式无分支 → 数值解打滑: " + fi.get("reason", "")
                fi["branches"] = 0
                return q, fi
            info["reason"] = "无分支（位姿不可达）"
            return None, info

        C = np.array(cand)
        q = C[int(np.argmin(np.linalg.norm(C - q_last, axis=1)))].copy()
        # ψ 细化：窄窗是离散采的，直接用格点会让臂角在相邻样点间来回跳。
        # 在选中的 ψ 附近反复折半再采，把台阶磨掉。
        if centre is not None:
            span = self.psi_span
            c = self.arm_angle(q)
            for _ in range(self.psi_refine):
                span *= 0.5
                more = self.branches(T_target, psi_centre=c, psi_span=span)
                if not more:
                    break
                Cm = np.array(more)
                k = int(np.argmin(np.linalg.norm(Cm - q_last, axis=1)))
                if np.linalg.norm(Cm[k] - q_last) < np.linalg.norm(q - q_last):
                    q = Cm[k].copy()
                    c = self.arm_angle(q)

        # 名义解 → 标定模型上牛顿细化
        for _ in range(self.refine_iters):
            e = self.cal.pose_error(self.cal.fk(q), T_target)
            if np.linalg.norm(e[:3]) < 1e-6 and np.linalg.norm(e[3:]) < 1e-6:
                break
            J = self.cal.jacobian(q)
            q = np.clip(q + np.linalg.lstsq(J, e, rcond=None)[0], self.lo, self.hi)

        d = q - q_last
        mx = float(np.max(np.abs(d)))
        if mx > self.max_step:
            q = q_last + d * (self.max_step / mx)
            info["reason"] = f"步长限幅 {mx:.3f}→{self.max_step}"
        e = self.cal.pose_error(self.cal.fk(q), T_target)
        info["pos_error"] = float(np.linalg.norm(e[:3]))
        info["rot_error"] = float(np.linalg.norm(e[3:]))
        self.stats["ok"] += 1
        return q, info
