#!/usr/bin/env python3
"""AR5 七轴的**构型判别**和**立体投影臂角**。照桌面 robot_kindyn/default/ikgeo 实现。

那份 C++ 里有两个我们原来没有的东西，都是「关节位姿合不合理」的关键：

1. **CFGX：三个比特描述臂处在哪个构型分支**（CSewCfgx.cpp）
   bit0 腕：J5 的符号
   bit1 肘：肘点在肩腕连线的哪一侧
   bit2 肩：腕心相对 J1 轴的前 / 后
   cfgx = 4×肩 + 2×肘 + 腕，取值 0..7

   这就是控制器区分构型的方式。**同一个末端位姿，不同 cfgx 是完全不同的臂型**
   （肘朝内 vs 朝外、肩在前 vs 在后），它们之间没法连续过渡，中间要经过奇异。
   原来我按连续代价挑解，会不知不觉跨过构型边界，臂就突然换个姿势 —— 这正是
   「关节位姿很奇怪」。正确的做法是**先筛出和参考同构型的候选，再在里面挑**。

   近零死区 5°：参考自己就贴在边界上时，这一位不参与判定（否则噪声会让它反复横跳）。

2. **立体投影 SEW 臂角**（CSewIk3rR3R.cpp 的 buildSewFrame）
   常规臂角是拿一个固定向量往肩腕连线的垂面上投影，肩腕连线和那个向量平行时
   投影退化，臂角会跳。立体投影的参考系用 (eSw − eT) × eR 构造，奇异只发生在
   肩腕连线正好指向 eT 那一条半线上，把 eT 选在工作空间用不到的方向就永远碰不到。
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9
DEADZONE_RAD = 0.0873          # 约 5°，和 C++ 的 kBitDeadzoneRad 一致


def _rot(h, th):
    h = np.asarray(h, dtype=float)
    K = np.array([[0, -h[2], h[1]], [h[2], 0, -h[0]], [-h[1], h[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


class SewConfig:
    """把 AR5PoeIK 的 POE 常数转成 H/P 两张表，再算构型和臂角。

    H: 3×7 关节轴方向（基座系，零位）
    P: 3×8 相邻关节原点之间的偏置，P[:,i] = r[i] − r[i-1]，r[-1] 取基座原点
    这和 C++ 里 SPoeKin 的约定一致，正解就是 Π exp(q_i ĥ_i) 逐段累加 P。
    """

    def __init__(self, base, e_t=None, e_r=None):
        self.base = base
        self.H = np.stack([np.asarray(w, dtype=float) for w in base.w], axis=1)   # 3×7
        r = [np.asarray(x, dtype=float) for x in base.r]
        P = [r[0]]
        for i in range(1, 7):
            P.append(r[i] - r[i - 1])
        P.append(np.zeros(3))
        self.P = np.stack(P, axis=1)                                             # 3×8
        # 立体投影的两个方向。eT 是奇异半线，选在工作空间够不到的方向（这里取 −Z，
        # 臂不会把腕心放到肩正下方）；eR 是参考方向，和 eT 不共线即可。
        self.e_t = np.asarray(e_t if e_t is not None else [0.0, 0.0, -1.0], dtype=float)
        self.e_r = np.asarray(e_r if e_r is not None else [0.0, 1.0, 0.0], dtype=float)
        self.e_t /= np.linalg.norm(self.e_t)
        self.e_r /= np.linalg.norm(self.e_r)
        # 肩/肘/腕在关节原点里的下标：由零位时和 AR5PoeIK 的 p_s / p_w0 对齐求出，
        # 不写死（换臂或换 DH 表时自己认）
        _, _, o0 = base._nom._chain(np.zeros(7))
        o0 = [np.asarray(x, dtype=float) for x in o0]
        self.i_s = int(np.argmin([np.linalg.norm(x - base.p_s) for x in o0]))
        self.i_w = int(np.argmin([np.linalg.norm(x - base.p_w0) for x in o0]))
        self.i_e = int((self.i_s + self.i_w) // 2)      # 肘在肩腕之间

    # ── 肩肘腕三点 ────────────────────────────────────────────────────────
    def sew_points(self, q):
        """肩、肘、腕三点在基座系下的位置。

        C++ 那份按 P 列累加、取第 3/4/5 个原点，那是它那条链的索引。**我们这条链不同**：
        实测零位下肩心在第 1 个关节原点（r[1]，与 AR5PoeIK.p_s 一致）、腕心在第 5 个
        （r[5]，与 p_w0 一致），肘在第 3 个。所以直接用运动链给出的各关节原点取索引，
        不重新累加，避免索引假设出错（第一版取 r[4] 当腕心，差了 278mm）。
        """
        q = np.asarray(q, dtype=float).reshape(7)
        _, _, origins = self.base._nom._chain(q)
        o = [np.asarray(x, dtype=float) for x in origins]
        return o[self.i_s].copy(), o[self.i_e].copy(), o[self.i_w].copy()

    # ── 构型三比特 ────────────────────────────────────────────────────────
    def bits(self, q):
        """返回 (腕, 肘, 肩) 三个 0/1。照 C++ calcBits。"""
        q = np.asarray(q, dtype=float).reshape(7)
        if not np.all(np.isfinite(q)):
            return None
        S, E, W = self.sew_points(q)

        wrist = 1 if q[4] < 0.0 else 0                    # bit0：J5 符号

        # bit1 肘：肘点相对肩腕连线的侧向
        sw = W - S
        n_sw = float(np.linalg.norm(sw))
        elbow = 0
        if n_sw > EPS:
            e_sw = sw / n_sw
            se = E - S
            se_perp = se - float(se @ e_sw) * e_sw
            ref = np.cross(e_sw, self.H[:, 0])
            if np.linalg.norm(ref) < EPS:
                ref = np.cross(e_sw, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(ref) < EPS:
                ref = np.cross(e_sw, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(ref) > EPS and float(se_perp @ ref) < 0.0:
                elbow = 1
        else:
            elbow = 1 if q[3] < 0.0 else 0

        # bit2 肩：腕心相对 J1 轴的前 / 后
        h0 = self.H[:, 0]
        radial = W - S
        radial = radial - float(radial @ h0) * h0
        sh_ref = np.cross(h0, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(sh_ref) < EPS:
            sh_ref = np.cross(h0, np.array([1.0, 0.0, 0.0]))
        shoulder = 1 if (np.linalg.norm(sh_ref) > EPS and float(radial @ sh_ref) < 0.0) else 0

        return wrist, elbow, shoulder

    def cfgx(self, q):
        b = self.bits(q)
        return None if b is None else 4 * b[2] + 2 * b[1] + b[0]

    def same_as_ref(self, q_ref, q_cand):
        """候选和参考是不是同一个构型。参考贴在某一位的边界上时那一位放宽。"""
        br, bc = self.bits(q_ref), self.bits(q_cand)
        if br is None or bc is None:
            return False
        q_ref = np.asarray(q_ref, dtype=float)
        # 腕：参考 J5 近零时不判
        if abs(q_ref[4]) >= DEADZONE_RAD and br[0] != bc[0]:
            return False
        # 肘：肩腕接近共线时退化，用 q4 的死区代替
        S, E, W = self.sew_points(q_ref)
        n_sw = float(np.linalg.norm(W - S))
        if (n_sw > EPS or abs(q_ref[3]) >= DEADZONE_RAD) and br[1] != bc[1]:
            return False
        # 肩：腕心径向太小则不判
        h0 = self.H[:, 0]
        radial = W - S
        radial = radial - float(radial @ h0) * h0
        if np.linalg.norm(radial) >= EPS and br[2] != bc[2]:
            return False
        return True

    # ── 立体投影臂角 ──────────────────────────────────────────────────────
    def _sew_frame(self, p_sw):
        """照 C++ buildSewFrame：用 (eSw − eT) × eR 定参考轴，避开常规臂角的奇异。"""
        n = float(np.linalg.norm(p_sw))
        if n < EPS:
            return None
        e_sw = p_sw / n
        k_rt = np.cross(e_sw - self.e_t, self.e_r)
        k_x = np.cross(k_rt, p_sw)
        n_kx = float(np.linalg.norm(k_x))
        if n_kx < EPS:
            return None
        e_x = k_x / n_kx
        return e_x, np.cross(e_sw, e_x), e_sw

    def psi(self, q):
        """由关节角求立体投影臂角。照 C++ sewFwdKin 的逆过程。"""
        S, E, W = self.sew_points(q)
        fr = self._sew_frame(W - S)
        if fr is None:
            return None
        e_x, e_y, e_sw = fr
        se = E - S
        se_perp = se - float(se @ e_sw) * e_sw
        if np.linalg.norm(se_perp) < EPS:
            return None
        return float(np.arctan2(se_perp @ e_y, se_perp @ e_x))
