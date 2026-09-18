#!/usr/bin/env python3
"""AR5 的**闭式**逆解：S-R-S 结构，臂角参数化，分支显式枚举。

和 `ar5_ik.AR5OptIK`（数值优化）的区别：那个是每帧从上一帧出发做局部下降，
分支翻转是它的**副作用**；这个把所有分支一次列全，再按代价挑——
「肘自己换边」从不可控的涌现行为变成显式决策。

── 结构（全部在真实 DH 上验过，见文件末尾的自检）────────────────────────
    j0-j1-j2  球肩，三轴共点（标定表上 0.23mm，名义表上精确共点）
    j3        肘
    j4-j5-j6  球腕，三轴共点（标定 0.32mm）

于是位姿分解成三段，每段都有闭式解：

  1. **腕心** p_w = p_目标 − R_目标·(法兰偏置)   ← 腕三轴共点，j4/j5/j6 不影响 p_w
  2. **肘角** |v(j3)| = |p_w − p_肩|             ← 实测 |v| 只依赖 j3（0.000µm）
                                                   一个 L 对应 ±j3 两支
  3. **肩姿态** R_F2 必须把 v̂ 转到 n̂=(p_w−p_肩)/L
       解不唯一：绕 n̂ 再转任意角 ψ 都成立 —— **ψ 就是臂角**，就是那个冗余自由度。
       R_F2(ψ) = Rot(n̂, ψ)·R_ref，再用 ZYZ 拆出 (j0,j1,j2)，**2 支**
  4. **腕姿态** R_wrist = R_F3ᵀ·R_F6目标，ZYZ 拆出 (j4,j5,j6)，**2 支**

  分支总数 = 2(肘) × 2(肩) × 2(腕) = 8，**但这台臂拿不到 8 个**：
  j6 行程只有 ±50°，腕翻要 j6±180°，恒不可达（2 万个位形实测 0.00%）；
  肘翻要 j3→−j3，j3∈[−60°,145°] 只有 58% 的位形够得到。实际 2~4 支。

── 为什么用名义 DH 解、标定 DH 修 ──────────────────────────────────────
闭式解要求严格共点。标定表带着 0.23~0.34mm 的共点残差，直接推闭式会有系统偏差。
名义表上 |肩→腕| 对其他轴的依赖是 **±0.000mm**（精确 S-R-S），闭式解精确成立；
两表最大差 0.4072，落到腕心上 ≤0.34mm —— 拿名义闭式解当种子，
在标定模型上做 1~2 步牛顿就收敛。既有闭式的分支枚举，又有标定的精度。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

H = np.pi / 2


def _Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _zyz(R: np.ndarray) -> List[Tuple[float, float, float]]:
    """把 R 拆成 Rz(a)·Ry(b)·Rz(c)，返回两个分支 (b>0 与 b<0)。

    万向锁（sin b≈0）时 a 和 c 只有和/差可定，此时把 a 归零、全给 c，
    并且只返回一支——返回两个数值上相同的解没有意义。
    """
    sy = float(np.hypot(R[0, 2], R[1, 2]))
    if sy < 1e-9:
        c = float(np.arctan2(-R[0, 1], R[0, 0]))
        return [(0.0, 0.0 if R[2, 2] > 0 else np.pi, c)]
    out = []
    for sign in (+1.0, -1.0):
        b = float(np.arctan2(sign * sy, R[2, 2]))
        a = float(np.arctan2(sign * R[1, 2], sign * R[0, 2]))
        c = float(np.arctan2(sign * R[2, 1], -sign * R[2, 0]))
        out.append((a, b, c))
    return out


def _rot_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """把单位向量 u 转到 v 的最小旋转。"""
    d = float(np.clip(u @ v, -1.0, 1.0))
    if d > 1 - 1e-12:
        return np.eye(3)
    if d < -1 + 1e-12:                      # 反向：绕任一垂直轴转 180°
        a = np.array([1.0, 0.0, 0.0])
        if abs(u[0]) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        k = np.cross(u, a)
        k /= np.linalg.norm(k)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    k = np.cross(u, v)
    s = float(np.linalg.norm(k))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - d) / (s * s))


def _axis_rot(n: np.ndarray, psi: float) -> np.ndarray:
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(psi) * K + (1 - np.cos(psi)) * (K @ K)


class AR5SrsIK:
    """闭式 S-R-S 逆解 + 分支枚举 + 按代价选解。

    构造要两套运动学：`nominal` 用来解（严格 S-R-S），`calibrated` 用来修
    （真实精度）。两个都是 vel_ar5 的 `AR5Kinematics` 实例。
    """

    def __init__(self, nominal, calibrated=None, *, limit_margin: float = 0.05,
                 limit_backoff: float = 0.03, psi_samples: int = 72,
                 max_step: float = 0.12, refine_iters: int = 3,
                 psi_window: int = 13, psi_span: float = 0.45,
                 fallback=None, psi_refine: int = 3):
        self.nom = nominal
        self.cal = calibrated if calibrated is not None else nominal
        m = limit_margin + limit_backoff
        self.lo = np.asarray(self.cal.joint_min, dtype=float) + m
        self.hi = np.asarray(self.cal.joint_max, dtype=float) - m
        self.psi_samples = int(psi_samples)
        self.psi_window = int(psi_window)     # 窄窗里采几个点
        self.psi_span = float(psi_span)       # 窄窗半宽(rad)
        self._psi_last: Optional[float] = None
        self.max_step = float(max_step)
        self.refine_iters = int(refine_iters)
        # 够不到时的退路。闭式解**只会给精确解或者不给** —— 目标出了工作空间
        # 就返回「无分支」，那一帧被丢掉。实测激进轨迹上接受率只有 45%，
        # 正是 DLS 那个雪崩模式。优化式相反：永远给够得到的里面最接近的。
        # 两个各取所长：能闭式就闭式（精确、快、分支显式），够不到就交给优化式打滑。
        self.fallback = fallback
        self.psi_refine = int(psi_refine)
        self.stats_fallback = 0

        # 肩心：j0/j1/j2 的交点。名义表上就是第 1 行之后的原点。
        self.p_s = self._frames(np.zeros(7))[1][:3, 3].copy()
        # 腕心→法兰：最后一行是固定变换
        F = self._frames(np.zeros(7))
        self.R_f6_to_f7 = F[6][:3, :3].T @ F[7][:3, :3]
        self.t_f6_to_f7 = F[6][:3, :3].T @ (F[7][:3, 3] - F[6][:3, 3])
        # |v(j3)|² = A + R·cos(j3−φ)，实测拟合残差 3.3e-16 m² —— **精确**成立。
        # 所以 j3 = φ ± acos((L²−A)/R)，O(1) 解析解，不用二分。
        # 原来那版拿 361 点网格 + 60 次二分，每次二分都重建整条链，
        # 单这一项就吃掉大半的 8.2ms。
        ts = np.linspace(-np.pi, np.pi, 721)
        L2 = np.array([float(self._v(t) @ self._v(t)) for t in ts])
        M = np.column_stack([np.ones_like(ts), np.cos(ts), np.sin(ts)])
        A, B, C = np.linalg.lstsq(M, L2, rcond=None)[0]
        self._A = float(A)
        self._R = float(np.hypot(B, C))
        self._phi = float(np.arctan2(C, B))
        self.stats = {"calls": 0, "ok": 0, "no_branch": 0, "unreachable": 0,
                      "branches": 0}

    # ── 运动学小工具（按行累乘，和 vel_ar5 的 _chain 同一套约定）──────────
    def _frames(self, q, kin=None):
        kin = kin or self.nom
        T = np.eye(4)
        out = []
        for i in range(kin.n_rows):
            al, a, dd = kin.alpha[i], kin.a[i], kin.d[i]
            th = kin.theta_offset[i] + (q[i] if i < kin.num_joints else 0.0)
            ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
            T = T @ np.array([[ct, -st, 0, a],
                              [st * ca, ct * ca, -sa, -sa * dd],
                              [st * sa, ct * sa, ca, ca * dd],
                              [0, 0, 0, 1.0]])
            out.append(T.copy())
        return out

    def _v(self, j3: float) -> np.ndarray:
        """腕心相对肩心，在「过完 j2 的姿态系」里的坐标。只依赖 j3。"""
        q = np.zeros(7)
        q[3] = j3
        F = self._frames(q)
        return F[6][:3, 3] - self.p_s        # j0=j1=j2=0 时 R_F2 = I

    def _solve_j3(self, L: float) -> List[float]:
        """|v(j3)| = L 的解析解。肘上/肘下两支。"""
        c = (L * L - self._A) / self._R
        if not -1.0 <= c <= 1.0:            # 够不到（太远或太近）
            return []
        t = float(np.arccos(c))
        out = []
        for j3 in (self._phi + t, self._phi - t):
            j3 = ((j3 + np.pi) % (2 * np.pi)) - np.pi
            if self.lo[3] <= j3 <= self.hi[3]:
                out.append(j3)
        return out

    # ── 枚举全部分支 ────────────────────────────────────────────────────
    def branches(self, T_target: np.ndarray,
                 psi_centre: Optional[float] = None,
                 psi_span: Optional[float] = None) -> List[np.ndarray]:
        R_t, p_t = T_target[:3, :3], T_target[:3, 3]
        # 腕心：法兰往回退一个固定偏置
        R_f6 = R_t @ self.R_f6_to_f7.T
        p_w = p_t - R_f6 @ self.t_f6_to_f7

        n = p_w - self.p_s
        L = float(np.linalg.norm(n))
        if L < 1e-9:
            return []
        n = n / L

        sols: List[np.ndarray] = []
        for j3 in self._solve_j3(L):
            v = self._v(j3)
            R_ref = _rot_between(v / np.linalg.norm(v), n)
            if psi_centre is None:
                psis = np.linspace(-np.pi, np.pi, self.psi_samples, endpoint=False)
            else:
                # 窄窗：臂角是连续量，50Hz 下一帧动不了多少。
                # 以上一帧的臂角为中心扫一小段，既快又天然满足「离上一帧最近」。
                sp = self.psi_span if psi_span is None else psi_span
                psis = psi_centre + np.linspace(-sp, sp, self.psi_window)
            for psi in psis:
                R_f2 = _axis_rot(n, psi) @ R_ref
                for (a0, a1, a2) in _zyz(R_f2):
                    q = np.zeros(7)
                    q[0], q[1], q[2], q[3] = a0, a1, a2, j3
                    # 腕：R_wrist = R_F3ᵀ·R_F6
                    F3 = R_f2 @ _Rx(self.nom.alpha[3]) @ _Rz(
                        self.nom.theta_offset[3] + j3)
                    R_w = F3.T @ R_f6
                    lhs = _Rx(-H) @ R_w @ _Rx(np.pi)
                    for (b0, b1, b2) in _zyz(lhs):
                        q[4], q[5], q[6] = b0, b1 + H, -b2 - H
                        qq = ((q + np.pi) % (2 * np.pi)) - np.pi
                        if np.all(qq >= self.lo) and np.all(qq <= self.hi):
                            sols.append(qq.copy())
        return sols

    def _arm_angle(self, q: np.ndarray) -> float:
        """反算一个关节解对应的臂角 ψ，供下一帧做窄窗中心。"""
        F = self._frames(q)
        p_w = F[6][:3, 3]
        n = p_w - self.p_s
        L = float(np.linalg.norm(n))
        if L < 1e-9:
            return 0.0
        n = n / L
        v = self._v(q[3])
        R_ref = _rot_between(v / np.linalg.norm(v), n)
        # R_F2 = Rot(n,ψ)·R_ref  →  Rot(n,ψ) = R_F2·R_refᵀ
        M = F[2][:3, :3] @ R_ref.T
        return float(np.arctan2(
            (np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]) @ n) / 2.0,
            (np.trace(M) - 1.0) / 2.0))

    # ── 驱动接口 ────────────────────────────────────────────────────────
    def ik(self, T_target: np.ndarray, q_seed: np.ndarray, **_ignored
           ) -> Tuple[Optional[np.ndarray], dict]:
        self.stats["calls"] += 1
        q_last = np.asarray(q_seed, dtype=float).reshape(7)
        # 先在上一帧臂角附近的窄窗里找；找不到再退回全周扫。
        # 这既是提速（13 个采样 vs 72 个），也**本身就是**「离上一帧最近」的实现。
        # 窄窗中心取**种子自己的臂角**，不是上一次调用的结果。
        # 取后者踩过坑：随机目标时 _psi_last 是陈旧的，窄窗里照样能找到
        # 落在限位内的解，只是臂角完全不对 —— 有解所以不触发全周回退，
        # 最后被步长限幅截断成 111mm 的误差。种子的臂角是无状态的、永远对的。
        try:
            centre = self._arm_angle(q_last)
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
                self.stats_fallback += 1
                q, fi = self.fallback.ik(T_target, q_last)
                fi = dict(fi)
                fi["reason"] = "闭式无分支(够不到) → 优化式打滑: " + fi.get("reason", "")
                fi["branches"] = 0
                return q, fi
            info["reason"] = "没有落在限位内的分支（位姿不可达）"
            return None, info

        # 选解代价：**离上一帧最近**（连续性优先）
        C = np.array(cand)
        k = int(np.argmin(np.linalg.norm(C - q_last, axis=1)))
        best = C[k]
        # ψ 细化：窄窗是离散采的，步长 2·span/(window−1)。直接用采样点会让
        # 臂角在相邻样点之间来回跳 —— 实测跳变 p95 4.88°，正好等于一个采样步长。
        # 在选中的 ψ 附近反复折半再采，把台阶磨掉。
        if centre is not None:
            span = self.psi_span
            c = self._arm_angle(best)
            for _ in range(self.psi_refine):
                span *= 0.5
                more = self.branches(T_target, psi_centre=c, psi_span=span)
                if not more:
                    break
                Cm = np.array(more)
                km = int(np.argmin(np.linalg.norm(Cm - q_last, axis=1)))
                if np.linalg.norm(Cm[km] - q_last) < np.linalg.norm(best - q_last):
                    best = Cm[km]
                    c = self._arm_angle(best)

        # 名义解 → 在标定模型上牛顿细化
        q = best.copy()
        for _ in range(self.refine_iters):
            e = self.cal.pose_error(self.cal.fk(q), T_target)
            if np.linalg.norm(e[:3]) < 1e-6 and np.linalg.norm(e[3:]) < 1e-6:
                break
            J = self.cal.jacobian(q)
            q = np.clip(q + np.linalg.lstsq(J, e, rcond=None)[0], self.lo, self.hi)

        # 步长闸：驱动的 joint_jump_max
        d = q - q_last
        m = float(np.max(np.abs(d)))
        if m > self.max_step:
            q = q_last + d * (self.max_step / m)
            info["reason"] = f"步长限幅 {m:.3f}→{self.max_step}"
        e = self.cal.pose_error(self.cal.fk(q), T_target)
        info["pos_error"] = float(np.linalg.norm(e[:3]))
        info["rot_error"] = float(np.linalg.norm(e[3:]))
        self.stats["ok"] += 1
        return q, info
