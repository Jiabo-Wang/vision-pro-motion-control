#!/usr/bin/env python3
"""AR5 七轴逆解的**选解层**：枚举全部解，按显式姿态代价挑一个。

架构照 IK-Geo（Elias & Wen, *IK-Geo: Unified Robot Inverse Kinematics Using
Subproblem Decomposition*）：冗余臂的做法是**固定多出来的那个自由度，把剩下 6 轴
用子问题闭式解出来，再扫那个自由度**；求解器返回**全部解**，由上层按自己的准则挑。
（ik_geo 的 Python 包只实现了 6 轴机型，7 轴要自己套这一层，所以这里只借架构。）

AR5PoeIK 已经把「扫臂角 ψ + 子问题闭式解」做好了，缺的就是选解策略：
它只在种子臂角附近 ±0.40rad 的窄窗里采样，再挑**离上一帧最近**的那个。
后果是臂角只会慢慢漂，一旦漂到别扭的位形（肘翻到内侧、腕关节贴着限位）就再也
回不来，因为没有任何「这个位形好不好」的评判 —— 这就是「关节位姿很奇怪」。

这里换成：全周扫 ψ → 每个候选算一个代价 → 取最小。代价四项：

  连续性   ‖q − q_seed‖                     不连续会让臂抽动，权重最大
  限位余量 Σ max(0, |用掉的行程| − 阈值)²    贴着限位的位形就是「别扭」的主因
  奇异     1/可操作度                        近奇异时小的末端移动要巨大的关节运动
  臂角偏好 (ψ − ψ_ref)²                      ψ_ref 取起始姿态的臂角，肘不会自己换边

前三项都是客观的，第四项是让臂保持在人看得懂的位形。权重可调，默认值在
真机上按「和控制器自带逆解的位形接近」定。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class AR5SelectiveIK:
    """包在 AR5PoeIK 外面，只换选解策略，闭式解本身不动。

    接口和 AR5PoeIK 一致：ik(T_target, q_seed) -> (q, info)，另外带 fk()。
    """

    def __init__(self, base=None, *, joint_min=None, joint_max=None,
                 w_cont: float = 1.0, w_limit: float = 4.0, w_sing: float = 0.02,
                 w_psi: float = 0.15, limit_soft: float = 0.75,
                 psi_samples: int = 48, psi_local: float = 0.5, psi_local_n: int = 9,
                 sweep_chunk: int = 0, sweep_trigger: float = 0.97, max_jump: float = 0.25,
                 max_step: float = 0.0148, **base_kw):
        if base is None:
            from ar5_poe_ik import AR5PoeIK
            base = AR5PoeIK(max_step=max_step, **base_kw)
        self.base = base
        self.w_cont, self.w_limit = float(w_cont), float(w_limit)
        self.w_sing, self.w_psi = float(w_sing), float(w_psi)
        self.limit_soft = float(limit_soft)
        self.psi_samples = int(psi_samples)
        self.psi_local, self.psi_local_n = float(psi_local), int(psi_local_n)
        self.sweep_chunk, self.sweep_trigger = int(sweep_chunk), float(sweep_trigger)
        self.max_jump = float(max_jump)
        self._sweep_i = 0
        self._since_sweep = 0
        self.lo = np.asarray(joint_min if joint_min is not None else base.lo, dtype=float)
        self.hi = np.asarray(joint_max if joint_max is not None else base.hi, dtype=float)
        self._mid = (self.lo + self.hi) / 2.0
        self._half = np.maximum((self.hi - self.lo) / 2.0, 1e-9)
        self.psi_ref: Optional[float] = None       # 偏好的臂角，接合时设一次
        self.q_ref_cfg = None                      # 构型参考姿态，接合时设一次
        try:
            from ar5_sew import SewConfig
            self.sew = SewConfig(self.base)
        except Exception:                          # noqa: BLE001
            self.sew = None
        self.stats = {"calls": 0, "ok": 0, "no_branch": 0, "full_sweep": 0,
                      "probe": 0, "cand": 0, "psi_drift": 0.0,
                      "cfg_filtered": 0, "cfg_lost": 0, "jump_rejected": 0}
        self._last_psi: Optional[float] = None

    # ── 给上层用 ──────────────────────────────────────────────────────────
    def fk(self, q):
        return self.base.fk(q)

    def anchor(self, q):
        """接合时调一次：锁定臂角和构型，之后臂就保持在这个臂型里。"""
        q = np.asarray(q, dtype=float)
        self.q_ref_cfg = q.copy()
        try:
            self.psi_ref = float(self.base.arm_angle(q))
        except Exception:                          # noqa: BLE001
            self.psi_ref = None
        self._last_psi = self.psi_ref
        if self.sew is not None:
            try:
                self.cfg_ref = self.sew.cfgx(q)
                self.psi_sew = self.sew.psi(q)
            except Exception:                      # noqa: BLE001
                self.cfg_ref = self.psi_sew = None
        return self.psi_ref

    # ── 代价 ──────────────────────────────────────────────────────────────
    def _cheap_cost(self, C, q_seed):
        """连续性 + 限位余量，全向量化，对所有候选算。"""
        d = C - q_seed
        cont = np.sum(d * d, axis=1)
        use = np.abs((C - self._mid) / self._half)          # 0=中位，1=贴限位
        over = np.maximum(use - self.limit_soft, 0.0)
        return self.w_cont * cont + self.w_limit * np.sum(over * over, axis=1)

    def _cost(self, C, q_seed, top=8):
        """先用便宜项筛出前 top 个，只对它们算臂角/奇异（要跑运动链，贵）。"""
        cheap = self._cheap_cost(C, q_seed)
        if (self.w_psi <= 0 or self.psi_ref is None) and self.w_sing <= 0:
            return cheap
        idx = np.argsort(cheap)[:max(1, min(top, len(C)))]
        cost = cheap.copy()
        for i in idx:
            q = C[i]
            if self.w_psi > 0 and self.psi_ref is not None:
                try:
                    dpsi = _wrap_pi(self.base.arm_angle(q) - self.psi_ref)
                    cost[i] += self.w_psi * dpsi * dpsi
                except Exception:                  # noqa: BLE001
                    pass
            if self.w_sing > 0 and hasattr(self.base, "jacobian"):
                try:
                    J = self.base.jacobian(q)
                    m = float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))
                    cost[i] += self.w_sing / max(m, 1e-4)
                except Exception:                  # noqa: BLE001
                    pass
        cost[np.setdiff1d(np.arange(len(C)), idx)] = np.inf   # 没算全的不参与竞争
        return cost

    # ── 主入口 ────────────────────────────────────────────────────────────
    def ik(self, T_target, q_seed, **_ignored) -> Tuple[Optional[np.ndarray], dict]:
        self.stats["calls"] += 1
        q_seed = np.asarray(q_seed, dtype=float).reshape(7)

        # 1) 搜索窗口钉在**锚定的臂角**上，不是跟着上一帧漂。
        #    原实现以种子臂角为中心，窗口每帧跟着解走 —— 于是臂角做随机游走，
        #    画一个圆下来漂 27°，肘一路游移，看着就是「位姿很奇怪」。
        #    钉住之后漂移≈0，而且成本和原来一样（还是一个窄窗）。
        cand = []
        centre = self.psi_ref
        if centre is None:
            centre = self._last_psi
        if centre is None:
            try:
                centre = float(self.base.arm_angle(q_seed))
            except Exception:                      # noqa: BLE001
                centre = None
        if centre is not None:
            cand = self.base.branches(T_target, psi_centre=centre, psi_span=self.psi_local)
            if not cand:                           # 钉住的臂角在这个位姿上无解：放宽
                cand = self.base.branches(T_target, psi_centre=centre,
                                          psi_span=self.psi_local * 3.0)

        # 2) 全局探索：窄窗只能就近跟随，永远发现不了「另一边有个好得多的位形」，
        #    臂角于是一路漂到别扭处。但一次全周扫要 20ms+，8ms 周期放不下，
        #    所以**分摊**：每次只取全周网格里的几个点，轮着来，十来个周期覆盖一遍。
        #    单次成本可控，全局性不丢。
        if self.sweep_chunk > 0:
            grid = np.linspace(-np.pi, np.pi, self.psi_samples, endpoint=False)
            take = min(self.sweep_chunk, self.psi_samples)
            idx = (self._sweep_i + np.arange(take)) % self.psi_samples
            self._sweep_i = int((self._sweep_i + take) % self.psi_samples)
            for psi_g in grid[idx]:
                cand += self.base.branches(T_target, psi_centre=float(psi_g),
                                           psi_span=float(np.pi / self.psi_samples))
            self.stats["probe"] += take
        # 局部解已经贴到限位了：立刻补一次完整扫描，别等分摊轮完
        if cand:
            Cl = np.clip(np.asarray(cand, dtype=float), self.lo, self.hi)
            worst = float(np.min(np.max(np.abs((Cl - self._mid) / self._half), axis=1)))
            if worst > self.sweep_trigger:
                old_n = getattr(self.base, "psi_samples", None)
                try:
                    if old_n is not None:
                        self.base.psi_samples = self.psi_samples
                    cand = list(cand) + list(self.base.branches(T_target))
                    self.stats["full_sweep"] += 1
                finally:
                    if old_n is not None:
                        self.base.psi_samples = old_n
        self._since_sweep = 0 if self.stats["full_sweep"] else self._since_sweep + 1

        if not cand:
            self.stats["no_branch"] += 1
            return None, {"branches": 0, "reason": "无分支（位姿不可达）"}

        C = np.asarray(cand, dtype=float)
        C = C[np.all(np.isfinite(C), axis=1)]
        C = np.clip(C, self.lo, self.hi)
        self.stats["cand"] += len(C)

        # 跳变门：先扔掉离上一帧太远的候选。同一构型里也有离得很远的解（差一个臂角
        # 半圈），而限位代价可能盖过连续性代价，实测出现过 1.55rad 的跳变。
        # 这道门是硬的：只有一个候选都不剩时才放开。
        if self.max_jump > 0 and len(C) > 1:
            near = np.max(np.abs(C - q_seed), axis=1) <= self.max_jump
            if near.any():
                self.stats["jump_rejected"] += int((~near).sum())
                C = C[near]

        # 构型筛选：先只留和参考同 CFGX 的候选（照 robot_kindyn/CSewCfgx）。
        # 同一个末端位姿，不同构型是完全不同的臂型（肘朝内/朝外、肩前/后），
        # 它们之间没法连续过渡，中间要经过奇异。只按连续代价挑会不知不觉跨过
        # 构型边界，臂就突然换个姿势 —— 这正是「关节位姿很奇怪」。
        # 实测：一个位姿的 21 个分支分属 2 个构型，筛完剩 10 个。
        if self.sew is not None and self.q_ref_cfg is not None and len(C) > 1:
            keep = np.array([self.sew.same_as_ref(self.q_ref_cfg, q) for q in C], dtype=bool)
            if keep.any():
                if not keep.all():
                    self.stats["cfg_filtered"] += int((~keep).sum())
                C = C[keep]
            else:
                # 同构型里一个解都没有：只能换构型。这时**只按连续性挑**（别让限位
                # 代价把臂甩到老远），挑完把构型参考更新过去，免得下一帧又来一次。
                self.stats["cfg_lost"] += 1
                k = int(np.argmin(np.sum((C - q_seed) ** 2, axis=1)))
                q_new = C[k].copy()
                self.q_ref_cfg = q_new.copy()
                try:
                    self.cfg_ref = self.sew.cfgx(q_new)
                except Exception:              # noqa: BLE001
                    pass
                C = q_new.reshape(1, -1)

        cost = self._cost(C, q_seed)
        k = int(np.argmin(cost))
        q = C[k].copy()

        # 3) 只在刚做过全周扫时细化一次（平时局部窗本身就够细）
        try:
            psi = float(self.base.arm_angle(q))
            if self.psi_local_n > 0:
                more = self.base.branches(T_target, psi_centre=psi,
                                          psi_span=self.psi_local / max(self.psi_local_n, 1))
                if more:
                    Cm = np.clip(np.asarray(more, dtype=float), self.lo, self.hi)
                    cm = self._cost(Cm, q_seed)
                    if cm.min() < cost[k]:
                        q = Cm[int(np.argmin(cm))].copy()
                        psi = float(self.base.arm_angle(q))
            if self._last_psi is not None:
                self.stats["psi_drift"] += abs(_wrap_pi(psi - self._last_psi))
            self._last_psi = psi
        except Exception:                          # noqa: BLE001
            psi = float("nan")

        self.stats["ok"] += 1
        use = float(np.max(np.abs((q - self._mid) / self._half)))
        return q, {"branches": len(C), "cost": float(cost[k]), "psi": psi,
                   "limit_use": use, "reason": ""}

    def report(self) -> str:
        s = self.stats
        n = max(1, s["calls"])
        cfg = getattr(self, "cfg_ref", None)
        return (f"选解: 调用 {s['calls']}  无解 {s['no_branch']}  平均候选 {s['cand']/n:.0f} 个/次  "
                f"臂角漂移 {np.rad2deg(s['psi_drift']):.0f}°  "
                f"构型滤掉 {s['cfg_filtered']} 个候选，被迫换构型 {s['cfg_lost']} 次"
                + (f"  (锚定 cfgx={cfg})" if cfg is not None else ""))


def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi
