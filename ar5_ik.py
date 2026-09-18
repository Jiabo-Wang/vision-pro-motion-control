#!/usr/bin/env python3
"""AR5 的改进版逆解 —— Project_Atom 自用，包在 vel_ar5 的 AR5Kinematics 外面。

不改 vel_ar5：FK / Jacobian / pose_error 全部委托给它原来的对象，只重写 `ik()`。
驱动里只有一个调用点（rokae_arm.py:847 `self.kinematics.ik(T, seed)`），
把 `env.arm._kinematics` 换成本类的实例即可。

相对原实现改了四处，每一处都对着实机上观察到的具体毛病：

1. **自适应阻尼**（原来是常数 λ=0.02）
   参考文献原话：「A constant λ value is inadequate for good performance over
   the arm's entire workspace. Low damping far from singular configurations
   and high damping close to a singular value.」
       λ² = 0                          当 σ_min ≥ ε
       λ² = λ_max²·(1 − (σ_min/ε)²)    否则
   远离奇异时几乎不加阻尼 → 更准更快收敛；接近奇异时阻尼自动升上来 → 不发散。
   常数阻尼是「两头都不讨好」：平时精度被压，真到奇异又不够。

2. **Weighted Least Norm 限位规避**（Chan & Dubey），替掉原来的零空间中位拉。
       V    = Σ (1/4)(θmax−θmin)² / [(θmax−θ)(θ−θmin)]
       ∂V/∂θ= (θmax−θmin)²(2θ−θmax−θmin) / [4(θmax−θ)²(θ−θmin)²]
       w    = 1 + |∂V/∂θ|   **仅当该关节正朝限位移动**
       w    = 1             否则
       Δθ   = W⁻¹Jᵀ(JW⁻¹Jᵀ + λ²I)⁻¹ e
   关键是那个「仅当朝限位移动」：原来的零空间中位拉是无差别的，
   人往回拽的时候它也在拖后腿。WLN 只在你往墙上撞的时候加阻力。
   这正对 j5/j6——它们行程只有 ±0.8727 rad，比别的轴窄 3.5 倍，
   实机上反复出现「j5 顶死→IK 全盘失败→臂冻住」。

3. **任务空间误差钳位**：每次迭代把 e 的平移/旋转分量各自限幅。
   目标很远时不钳位会让第一步巨大，直接把种子甩飞、落到别的解支上。

4. **早退**：误差连续若干次不再下降就立刻放弃，不把 100 次迭代跑满。
   实机日志里 IK 失败时每帧都烧满 100 次迭代（每次一个 7×6 Jacobian +
   线性求解），把 50Hz 主循环拖到 43Hz，手也跟着卡——这是「反解很慢」的
   直接来源。失败要趁早，省下的时间留给下一帧。

────────────────────────────────────────────────────────────────────────────
本文件里有**两个**求解器。上面那个 `AR5IK` 是雅可比迭代（DLS）路线，
下面的 `AR5OptIK` 是**优化**路线，默认用后者。原因见 AR5OptIK 的文档。
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np


class AR5IK:
    """包住原来的 AR5Kinematics，只换掉 ik()。其余属性透传。"""

    def __init__(self, base, *, lambda_max: float = 0.08, sigma_eps: float = 0.04,
                 wln_gain: float = 1.0, max_pos_err: float = 0.05,
                 max_rot_err: float = 0.35, stall_patience: int = 6,
                 limit_margin: float = 0.05):
        self._base = base
        self.lambda_max = lambda_max      # 奇异点处的阻尼上限
        self.sigma_eps = sigma_eps        # 「奇异区」有多大
        self.wln_gain = wln_gain          # WLN 权重强度，0 = 退化成普通 DLS
        self.max_pos_err = max_pos_err    # 单次迭代的平移误差上限(米)
        self.max_rot_err = max_rot_err    # 单次迭代的旋转误差上限(弧度)
        self.stall_patience = stall_patience
        # ⚠ 按**软限位**求解，不是原始限位。
        # vel_ar5 的 kinematics 用 joint_min/max 夹解，而 rokae_arm 的接受判据是
        # joint_min/max ± joint_limit_margin(0.05)。于是 IK 会交出结构上
        # 永远通不过的解（实机报 j5=-0.8727 outside [-0.8227,0.8227]，
        # 基准里两个求解器的「最紧余量」都正好是 -2.9° = -0.05rad）。
        # 在这里就按驱动的带子求解，交出去的每个解都是可接受的。
        self.soft_lo = np.asarray(base.joint_min, dtype=float) + limit_margin
        self.soft_hi = np.asarray(base.joint_max, dtype=float) - limit_margin
        # 统计，跑完一轮可以看
        self.stats = {"calls": 0, "ok": 0, "stalled": 0, "limits": 0,
                      "iters": 0, "damped": 0}

    # 没重写的属性一律透传给原对象（fk / jacobian / joint_min / manipulability ...）
    def __getattr__(self, name):
        return getattr(self._base, name)

    # ── 各个零件 ────────────────────────────────────────────────────────────
    def _clamp_error(self, e: np.ndarray) -> np.ndarray:
        """限幅任务空间误差。平移和旋转分开限，各自保方向。"""
        e = e.copy()
        for sl, cap in ((slice(0, 3), self.max_pos_err), (slice(3, 6), self.max_rot_err)):
            n = float(np.linalg.norm(e[sl]))
            if n > cap:
                e[sl] *= cap / n
        return e

    def _damping_sq(self, J: np.ndarray) -> float:
        """按最小奇异值自适应给阻尼平方。远离奇异时返回 0。"""
        try:
            sigma_min = float(np.linalg.svd(J, compute_uv=False)[-1])
        except np.linalg.LinAlgError:
            return self.lambda_max ** 2
        if sigma_min >= self.sigma_eps:
            return 0.0
        self.stats["damped"] += 1
        return (self.lambda_max ** 2) * (1.0 - (sigma_min / self.sigma_eps) ** 2)

    def _wln_weights(self, q: np.ndarray, dq_prev: Optional[np.ndarray]) -> np.ndarray:
        """Chan & Dubey 的限位规避权重。**只惩罚朝限位移动的关节。**"""
        lo, hi = self.soft_lo, self.soft_hi
        rng = np.maximum(hi - lo, 1e-9)
        # 夹一点点余量，免得正好贴边时分母炸
        qc = np.clip(q, lo + 1e-4 * rng, hi - 1e-4 * rng)
        num = (rng ** 2) * (2.0 * qc - hi - lo)
        den = 4.0 * ((hi - qc) ** 2) * ((qc - lo) ** 2)
        dV = num / np.maximum(den, 1e-12)
        w = np.ones_like(q)
        if dq_prev is None:
            moving_to_limit = np.ones_like(q, dtype=bool)   # 第一次迭代没有历史，保守起见全算
        else:
            moving_to_limit = (dV * dq_prev) > 0            # 梯度与运动同号 = 在往限位走
        w[moving_to_limit] += self.wln_gain * np.abs(dV[moving_to_limit])
        return w

    # ── 主流程 ──────────────────────────────────────────────────────────────
    def ik(self, T_target: np.ndarray, q_seed: np.ndarray, max_iters: int = 100,
           pos_tol: float = 1e-4, rot_tol: float = 1e-3, **_ignored
           ) -> Tuple[Optional[np.ndarray], dict]:
        base = self._base
        n = base.num_joints
        lo, hi = self.soft_lo, self.soft_hi
        q = np.asarray(q_seed, dtype=np.float64).reshape(n).copy()
        self.stats["calls"] += 1

        info = {"reason": "", "iters": 0, "pos_error": np.inf, "rot_error": np.inf}
        best_err, stall = np.inf, 0
        dq_prev = None

        for it in range(max_iters):
            T = base.fk(q)
            e_full = base.pose_error(T, T_target)
            pos_err = float(np.linalg.norm(e_full[:3]))
            rot_err = float(np.linalg.norm(e_full[3:]))
            info.update(iters=it + 1, pos_error=pos_err, rot_error=rot_err)
            self.stats["iters"] += 1

            if pos_err < pos_tol and rot_err < rot_tol:
                if np.any(q < lo) or np.any(q > hi):
                    # 冗余臂通常在限位内也有解，先夹进范围再验一次
                    q_in = np.clip(q, lo, hi)
                    e_in = base.pose_error(base.fk(q_in), T_target)
                    if (np.linalg.norm(e_in[:3]) < pos_tol * 3
                            and np.linalg.norm(e_in[3:]) < rot_tol * 3):
                        info["reason"] = "converged after clamping into limits"
                        self.stats["ok"] += 1
                        return q_in, info
                    self.stats["limits"] += 1
                    info["reason"] = f"converged outside joint limits: {base.limit_report(q)}"
                    return None, info
                info["reason"] = "converged"
                self.stats["ok"] += 1
                return q, info

            # 早退：误差不再下降就别耗着（失败要趁早，省下的算力留给下一帧）
            total = pos_err + rot_err
            if total < best_err - 1e-7:
                best_err, stall = total, 0
            else:
                stall += 1
                if stall >= self.stall_patience:
                    self.stats["stalled"] += 1
                    info["reason"] = (f"stalled at pos={pos_err*1000:.1f}mm "
                                      f"rot={np.rad2deg(rot_err):.1f}° after {it+1} iters")
                    return None, info

            e = self._clamp_error(e_full)
            J = base.jacobian(q)
            lam2 = self._damping_sq(J)
            w = self._wln_weights(q, dq_prev)
            Winv = np.diag(1.0 / w)

            # Δθ = W⁻¹Jᵀ(J W⁻¹ Jᵀ + λ²I)⁻¹ e
            JWJt = J @ Winv @ J.T + lam2 * np.eye(6)
            try:
                dq = Winv @ J.T @ np.linalg.solve(JWJt, e)
            except np.linalg.LinAlgError:
                info["reason"] = "singular Jacobian"
                return None, info

            norm = float(np.linalg.norm(dq))
            if norm > 0.15:                     # 有界步长，别把种子甩飞
                dq *= 0.15 / norm
            dq_prev = dq
            q = np.clip(q + dq, lo, hi)

        info["reason"] = f"did not converge in {max_iters} iterations"
        return None, info


class AR5OptIK:
    """把逆解当**优化问题**解，而不是解方程。默认求解器。

    读 chf_ws（RoboTele，CasADi+Pinocchio+IPOPT）的 GenericDualArmSolver 之后
    改的。它和 DLS 路线最本质的区别只有一句话：

        **位姿误差是代价，不是约束；关节限位是约束，不是代价。**

    DLS 是反过来的——它要求解出来的 q 精确落在目标上（误差进 tol 才算数），
    限位则是事后 clip。于是目标够不到的时候它只能返回 None。而
    rokae_arm.py:847 拿到 None 就 `return False`，那一帧的目标**整个被丢掉**，
    臂没有新目标 → 保持原位 → 下一帧能解了再猛地跟上。
    「IK 不太顺」的手感就是这么来的：不是解得不准，是**时不时没有解**。

    优化式不会没有解。目标够不到，它就交出够得到的里面最接近的那个，
    臂平滑地顶在工作空间边界上等你回来——这正是遥操想要的「打滑」。

        min_q   wT·‖p(q)−p*‖² + wR·‖log(R(q)R*ᵀ)‖²
              + wS·‖q−q_last‖² + wReg·‖q−q_rest‖²
        s.t.    lo ≤ q ≤ hi                （硬约束，解出来必然合法）

    四项分别买到什么：

    * **wT : wR = 100 : 1**（chf_ws 四套机器人共用的比例，照搬）。
      这是个**汇率**：√(wT/wR)=10 rad/m，即 1 cm 位置 ≙ 5.7° 姿态。
      位置和姿态打架时优先保位置。对着实机那条
      「j5=-0.8571 outside [-0.8227,0.8227]」——腕部姿态够不到的时候，
      DLS 是整解失败，这里是**姿态让步、位置保住**，手腕差一点但臂在动。
      用户说的「手腕反解感觉不是很对」和「不太顺」大概率是同一件事的两面。

      **绝对值比 chf 放大了 10 倍**（50/0.5 → 500/5），这个不能照抄。
      任务权重和 wSmooth/wReg 的比值决定「稳态偏差」：手停住不动了，
      解出来的位姿离目标还差多少。照 chf 的 50/0.5 实测差 3.0°——
      用户本来就在抱怨手腕姿态不对，再吃 3° 说不过去。放大到 500/5
      降到 0.29°，代价只是每次多约 0.5ms。再往上（k=30）到 0.08°，
      但跳变和耗时都开始变差，10 倍是拐点。

    * **wS·‖q−q_last‖²** 平滑项，DLS 完全没有的东西。
      7 轴是冗余的，同一个 TCP 位姿对应一整族关节解。DLS 每帧独立求解，
      种子稍微一动就可能跳到另一个解支上——TCP 没怎么动，肘部却甩一下。
      平滑项把「和上一帧接近」写进目标函数，是**连续的**代价而不是
      DLS 那种硬步长上限（0.15）：该快的时候不限速，该稳的时候自己会稳。

    * **wReg·‖(q−q_rest)/range‖²** 防漂移，按归一化行程算（理由见 __init__）。
      权重开得很小：实测它并不能把 j5/j6 从限位上拉回来——7 轴的零空间
      只有 1 维（肘部绕肩腕连线转），腕部姿态是任务**完全决定**的，
      不是冗余自由度。wReg 从 0 加到 2.0，j5/j6 的行程占用一直是 94%，
      纹丝不动；能改变的只有稳态偏差和抖动，两者都是越小越好。
      真正管住零空间的其实是 wSmooth（「跟上一帧接近」本身就是连续性）。
      所以 wReg 只留一点点，用来防止长时间的缓慢漂移。

    * 限位和每帧步长都是 L-BFGS-B 的 box 约束，**解出来对驱动的两道闸
      （joint limits / joint_jump_max）天然合法**，不会再出现
      「解出来了但被驱动拒掉」。

    实测（合成 DH + 真实限位，1200 帧 50Hz 轨迹，含驱动的两道闸）：

        求解器              驱动接受率   跳变p95   ms中位
        DLS AR5IK            13.9%      —        1.5
        AR5OptIK            100.0%     1.36°     2.1     ← 温和轨迹
        DLS AR5IK             0.0%      —        3.7
        AR5OptIK            100.0%     3.18°     2.9     ← 故意超出工作空间

    DLS 那 13.9%/0% 不是笔误：它解不出来的帧丢掉，解出来的帧里又有一大半
    因为解支跳变被驱动拒掉（温和轨迹 1292/1500 帧栽在这一条）。
    而且一旦被拒，臂不动、目标继续跑，下一帧要跳得更远——会**雪崩**。
    这就是「IK 不太顺」的全貌。

    实现上不用 CasADi/Pinocchio——vel_ar5 已经有 fk/jacobian/pose_error，
    梯度可以直接写出来（见 `_cost_grad`），scipy 的 L-BFGS-B 够快。
    """

    def __init__(self, base, *, w_trans: float = 500.0, w_rot: float = 5.0,
                 w_smooth: float = 0.5, w_reg: float = 0.02,
                 limit_margin: float = 0.05, max_iter: int = 50,
                 tol: float = 1e-6, q_rest: Optional[np.ndarray] = None,
                 accept_pos: float = 0.10, accept_rot: float = 1.2,
                 max_step: float = 0.12, limit_backoff: float = 0.03,
                 lock: Optional[dict] = None,
                 w_limit: float = 4.0, barrier_start: float = 0.90,
                 stuck_pos: float = 0.010, stuck_rot: float = 0.26,
                 min_manip: float = 0.01):
        self._base = base
        self.w_trans, self.w_rot = float(w_trans), float(w_rot)
        self.w_smooth, self.w_reg = float(w_smooth), float(w_reg)
        self.max_iter, self.tol = int(max_iter), float(tol)
        # 求解带子要**严格窄于**驱动的带子，不能刚好相等。
        # 驱动的判据是 rokae_arm.py:158 `_joint_lo = joint_min + margin`——
        # 和这里一模一样。解正好落在边界上时由浮点舍入决定过不过，
        # config.py 给 joint_goal_step(0.075) < joint_jump_max(0.15) 写的
        # 那段注释记的就是这个坑：顶着阈值设，实测 100% 被拒、臂一动不动。
        self.soft_lo = np.asarray(base.joint_min, dtype=float) + limit_margin + limit_backoff
        self.soft_hi = np.asarray(base.joint_max, dtype=float) - limit_margin - limit_backoff
        # 锁轴：把某几个关节的上下界压成同一个值，box 约束天然实现「锁死」。
        # 锁掉一轴 = 7 轴降到 6 轴，冗余消失，零空间也就没了 —— 肘不会再自己换边。
        # 对 j5/j6 这种只有 ±50° 的窄轴尤其有用：锁在中位就永远不会顶限位。
        # 代价是任务空间少一个自由度，够不到的位姿变多（靠 best-effort 打滑兜底）。
        self.locked = dict(lock or {})
        for j, val in self.locked.items():
            v = float(np.clip(val, self.soft_lo[j], self.soft_hi[j]))
            self.soft_lo[j] = self.soft_hi[j] = v
        # 限位势垒用的中点/半行程（按**限位带子**算，不是每帧的步长带子）
        self._mid = 0.5 * (self.soft_lo + self.soft_hi)
        self._half = np.maximum(0.5 * (self.soft_hi - self.soft_lo), 1e-9)
        # 势垒**带死区**：行程用掉 barrier_start 之前代价严格为 0。
        # 第一版写的是全程 b^8，看着「中间几乎免费」，实测却贵得离谱：
        # 正常跟踪的姿态残差 0.24°→1.79°，j5/j6 可用行程 91%→69%，
        # 等于白白没收操作者四分之一的手腕行程。原因是 b=0.7 处势垒已经
        # 0.058，而 2° 姿态误差的代价才 0.006 —— 势垒在中段就压过任务了。
        # 死区版在 85% 之前完全不存在，只在最后 15% 立一堵墙。
        self.w_limit = float(w_limit)
        self.barrier_start = float(np.clip(barrier_start, 0.0, 0.999))
        self._b_span = 1.0 - self.barrier_start
        # 「跟不上」的判据（供上层重锚用），按实测残差分布定，不是拍的：
        # 温和轨迹残差 p99 = 1.6mm/4.4°，长期顶在工作空间边界时 p99 = 37.6mm/20.6°。
        # 取 10mm/15° 正好把两者分开。
        self.stuck_pos, self.stuck_rot = float(stuck_pos), float(stuck_rot)
        # 下发前的可达性校验：**奇异闸**（Yoshikawa 可操作度下限）。
        #
        # 为什么不能用驱动现成的 check_reachable：它走 `robot.checkPath`，是
        # 控制器往返，ar5_env.py 里量过 **42ms 且持 GIL** —— 50Hz 只有 20ms 预算，
        # 而且会把 1kHz 的 RT 发送线程饿死。那个函数本来就注明「绝不用在 RT 路径上」。
        # 所以这里用**本地**判据：一次 Jacobian + 一个 6×6 行列式，约 0.1ms。
        #
        # 挡什么：解在关节限位内、跳变也合法，但落在奇异附近——那里笛卡尔方向上
        # 一点点位移要极大的关节运动，伺服跟不上，跟随误差变成力矩尖峰，
        # 被控制器的碰撞检测判成撞了 → **硬停掉电抱闸**（用户说的「抱死」）。
        # 关节空间的闸拦不住这种，因为解本身完全合法。
        #
        # 机制是**减速**，不是拦截。
        # 第一版写的是「往奇异里钻就原地不动」，自检当场打脸：从奇异位形起步时
        # 40 帧全被拦、可操作度一动不动 —— **把自己锁死在奇异点上出不来**。
        # 因为求解器的代价函数里没有可操作度这一项，它不会主动往外走，
        # 「拦住变差的一步」就等于永远不动。
        # 改成按 w/min_manip 缩步长，并留 slow_floor 的地板：越接近奇异走得越慢
        # （伺服跟得上，不出力矩尖峰），但**永远保留移动能力**，所以一定走得出来。
        self.min_manip = float(min_manip)
        self.slow_floor = 0.15          # 最慢降到 15% 步长，不归零
        self._last_manip: Optional[float] = None
        self.q_rest = (np.asarray(base.joint_mid, dtype=float).copy()
                       if q_rest is None else np.asarray(q_rest, dtype=float).copy())
        np.clip(self.q_rest, self.soft_lo, self.soft_hi, out=self.q_rest)
        # 正则项按**归一化行程**算：Σ((q−rest)/range)²，不是 Σ(q−rest)²。
        # 用弧度算的话，j0/j2/j4（行程 6.2rad）在代价里天然比 j5/j6（1.75rad）
        # 重 12 倍，正则于是主要在推那几个本来就宽松的轴——既没管住该管的，
        # 又在零空间里把臂推来推去。实测（k=10 温和轨迹）：
        # 归一化前跳变 p95 2.61°，wReg 直接置 0 反而降到 1.33°，就是这个原因。
        # 归一化之后「离自己限位有多近」对每个轴是同一把尺子，
        # 行程只有 ±0.8727 的 j5/j6 自动被照顾到。
        self._reg_scale = 1.0 / np.maximum(self.soft_hi - self.soft_lo, 1e-9)
        # 这**不是**精度门槛，是防呆网，宁可放过不可错杀——错杀就是丢帧，
        # 正是这个求解器要消掉的毛病。
        # 阈值按实测的「合法打滑残差」定，不是拍的：两条缰绳（平移 4cm、
        # 姿态 0.35rad）已经把单帧能差多远卡死了，2000 帧长期顶在工作空间
        # 边界上跑，合法残差最大 39.3mm / 23.3°。取 100mm / 69° ≈ 2.5~3 倍余量。
        # 真正兜底的其实是下面的 max_step：再脏的目标，一帧也只能推动 0.12rad。
        self.accept_pos, self.accept_rot = float(accept_pos), float(accept_rot)
        # 每帧的单关节步长上限，**也做成 box 约束**。
        # 驱动第四道闸（rokae_arm.py:787）：max|q−q_cmd| > joint_jump_max(0.15rad)
        # 直接拒收，日志原话「This is what an inverse-kinematics branch flip
        # looks like」。冗余臂在零空间里翻解支的时候就会这样——TCP 只动了几毫米，
        # 肘部甩过去几十度。那一帧和解不出来一样，是**丢帧**。
        # 写进 box 之后，交出去的解在限位和跳变两道闸上都天然合法，
        # 而且够不到时会沿着上限匀速趋近，而不是跳一下被拒。
        # 0.12 < 0.15 留余量，理由和 config.py 里 joint_goal_step 的注释一样：
        # 顶着阈值设会让舍入决定成败。
        self.max_step = float(max_step)
        # `saturated` 是这个求解器**必须**往外报的东西。
        # avp_arm_teleop 的两张自救网都是以「IK 失败」为信号的：
        #   打滑重锚  ← env.stats["unreachable"]（只在 ik() 返回 None 时加）
        #   限位自救  ← arm.rt_stats()["rejected"]["limits"]（只在驱动拒收时加）
        # 而本求解器永不返回 None、交出的解也永远过得了驱动 ——
        # 于是**两个计数器永远是 0，两张网同时失效**。实测后果：腕部姿态
        # 够不到时 6 帧顶死 j5/j6，然后谁也不吭声，臂就那么卡着（用户报的现象）。
        # 所以「顶在限位上而且够不到」这件事要自己数出来，让上层能重锚。
        self.stats = {"calls": 0, "ok": 0, "rejected": 0, "nfev": 0,
                      "slipped": 0, "stepcap": 0, "saturated": 0, "singular": 0,
                      "seconds": 0.0}

    def __getattr__(self, name):
        return getattr(self._base, name)

    # ── 代价与梯度 ──────────────────────────────────────────────────────────
    def _fk_jac(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """一次 `_chain` 同时拿 FK 和 Jacobian。

        照着写 `base.fk(q)` + `base.jacobian(q)` 会跑两遍运动链，而每次
        代价求值都要这两样——白白翻倍。链本身是 Python 里的 7 次 4×4 连乘，
        正是这个求解器的热点。
        """
        base = self._base
        # ⚠ 包装类（如 ToolFrameKinematics）重写了 fk/jacobian，但 `_chain` 会被
        # __getattr__ 透传到**底层的法兰运动学**上 —— 直接抄近路就把包装绕过去了，
        # 解出来的仍然是法兰位姿。实测症状：工具系求解恒偏 147.7mm（正是工具偏置本身）。
        # 有 fk/jacobian 重写的一律走正门。
        if getattr(base, "_overrides_fk", False):
            return base.fk(q), base.jacobian(q)
        T, axes, origins = base._chain(q)
        p_end = T[:3, 3]
        n = base.num_joints
        J = np.empty((6, n))
        for i in range(n):
            J[:3, i] = np.cross(axes[i], p_end - origins[i])
            J[3:, i] = axes[i]
        return T, J

    def _cost_grad(self, q: np.ndarray, T_target: np.ndarray,
                   q_last: np.ndarray) -> Tuple[float, np.ndarray]:
        T, J = self._fk_jac(q)
        e = self._base.pose_error(T, T_target)          # target ⊖ current
        ep, er = e[:3], e[3:]
        dq_s = q - q_last
        dq_r = (q - self.q_rest) * self._reg_scale      # 归一化行程，见 __init__

        # 限位势垒：b∈[-1,1] 是「用掉了多少行程」，代价取 b^(2p)。
        # 中间几乎免费（b=0.5 时 0.004），越靠近限位越陡（b=0.9→0.43, b=1→1.0）。
        # 没有这一项的后果实测过：目标姿态够不到时，求解器会**主动**把
        # j5/j6 推到停止位上去榨那点旋转代价——6 帧(120ms)就顶死，然后卡在那里。
        # 二次的居中拉（w_reg）救不了：要么中间就太硬，要么到边上还太软。
        # 势垒的意思是「宁可姿态差一点，也别把腕子顶死」，这正是遥操要的取舍。
        b = (q - self._mid) / self._half                    # 用掉了多少行程, ∈[-1,1]
        exc = np.maximum(np.abs(b) - self.barrier_start, 0.0) / self._b_span
        exc3 = exc ** 3

        cost = (self.w_trans * float(ep @ ep) + self.w_rot * float(er @ er)
                + self.w_smooth * float(dq_s @ dq_s) + self.w_reg * float(dq_r @ dq_r)
                + self.w_limit * float((exc3 * exc).sum()))

        # ∂‖e‖²/∂q = −2eᵀJ，**平移和旋转都是精确的**，不是小角近似。
        # 平移显然。旋转那半是这么来的：记 φ=e_r=log(R_t R_cᵀ)，世界系角速度 ω=J_w q̇，
        # 则 d(R_t R_cᵀ)/dt 对应的左角速度是 −Eω（E=R_t R_cᵀ）。于是
        #     φ̇ = J_l⁻¹(φ)·(−Eω)
        # 而 J_l⁻¹(φ)φ = φ 且 Eφ = φ（E 就是绕 φ 转的），所以
        #     φᵀφ̇ = −φᵀEω = −φᵀω
        # 左乘的那些修正项全落在 φ 的正交补上，点到 φ 上正好消掉。
        # 实测：300 个随机姿态、旋转误差中位 77° 最大 178°，
        # 与数值差分的相对误差最大 3.4e-8 —— 机器精度，确实是精确的。
        grad = (-2.0 * (self.w_trans * (J[:3].T @ ep) + self.w_rot * (J[3:].T @ er))
                + 2.0 * self.w_smooth * dq_s
                + 2.0 * self.w_reg * dq_r * self._reg_scale      # 链式法则再乘一次
                + self.w_limit * 4.0 * exc3 * np.sign(b) / (self._b_span * self._half))
        return cost, grad

    # ── 主流程 ──────────────────────────────────────────────────────────────
    def ik(self, T_target: np.ndarray, q_seed: np.ndarray, max_iters: Optional[int] = None,
           pos_tol: float = 1e-4, rot_tol: float = 1e-3, **_ignored
           ) -> Tuple[Optional[np.ndarray], dict]:
        from scipy.optimize import minimize

        base = self._base
        # ⚠ 步长必须从**原始 seed** 量，不是从夹进求解带之后的 seed 量。
        # 驱动的跳变闸比的是 |q − _q_cmd|，_q_cmd 就是这个原始 seed。
        # 先夹再限步的话，seed 落在「求解带之外、驱动带之内」那条 backoff
        # 宽的夹缝里时，实际位移 = backoff(0.03) + max_step(0.12) = 0.15，
        # 正好等于 joint_jump_max —— 又回到舍入决定成败。离线自检抓到的就是这个。
        q_last = np.asarray(q_seed, dtype=np.float64).reshape(base.num_joints).copy()
        self.stats["calls"] += 1
        t0 = time.perf_counter()

        # 奇异减速：用**上一帧**解出来的可操作度定这一帧的步长（因果的，不用多跑链）
        step = self.max_step
        if self._last_manip is not None and self._last_manip < self.min_manip:
            step *= max(self.slow_floor, self._last_manip / max(self.min_manip, 1e-12))
            self.stats["singular"] += 1

        # 限位 ∩ 步长：解出来对驱动的两道闸（limits / jump）都天然合法
        lo = np.maximum(self.soft_lo, q_last - step)
        hi = np.minimum(self.soft_hi, q_last + step)
        # seed 已经在求解带之外时交集为空（臂本来就停在夹缝或带外）。
        # 那就朝带子方向走一步，步长照样受 max_step 管。
        empty = lo > hi
        if np.any(empty):
            toward = np.clip(np.clip(q_last, self.soft_lo, self.soft_hi),
                             q_last - self.max_step, q_last + self.max_step)
            lo = np.where(empty, toward, lo)
            hi = np.where(empty, toward, hi)
        x0 = np.clip(q_last, lo, hi)

        res = minimize(self._cost_grad, x0, args=(T_target, q_last),
                       method="L-BFGS-B", jac=True,
                       bounds=list(zip(lo.tolist(), hi.tolist())),
                       options={"maxiter": int(max_iters or self.max_iter),
                                "ftol": self.tol, "gtol": 1e-8})
        q = np.clip(np.asarray(res.x, dtype=np.float64), lo, hi)
        if np.max(np.abs(q - q_last)) > 0.95 * step:
            self.stats["stepcap"] += 1      # 顶着步长上限在走 = 臂在全速追

        # 一次 _fk_jac 同时拿位姿误差和可操作度，不额外跑运动链
        T_sol, J_sol = self._fk_jac(q)
        manip = float(np.sqrt(max(0.0, np.linalg.det(J_sol @ J_sol.T))))
        self._last_manip = manip            # 下一帧据此减速
        near_singular = manip < self.min_manip

        e = base.pose_error(T_sol, T_target)
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:]))
        self.stats["nfev"] += int(res.nfev)
        self.stats["seconds"] += time.perf_counter() - t0

        # 「跟不上」：残差明显超出正常跟踪水平。这就是上层重锚要的信号。
        #
        # 第一版判据写的是「顶到限位 且 有残差」，结果是**死代码**：
        # 限位势垒一加上，关节根本不会再顶到限位，那个条件永远不成立。
        # 判据必须盯**症状**（跟不上），不是盯某一种成因（顶限位）——
        # 够不到就是够不到，臂是顶在关节限位上还是顶在工作空间边界上，
        # 对操作者是同一件事。
        b_use = float(np.max(np.abs((q - self._mid) / self._half)))
        # 近奇异的帧一律算「跟不上」：臂被减速了、人手还在全速走，
        # 对应关系正在拉开，必须让上层重锚。
        stuck = bool(pos_err > self.stuck_pos or rot_err > self.stuck_rot
                     or near_singular)
        if stuck:
            self.stats["saturated"] += 1

        info = {"iters": int(res.nit), "nfev": int(res.nfev),
                "pos_error": pos_err, "rot_error": rot_err, "reason": "",
                "saturated": stuck, "travel_used": b_use,
                "manipulability": manip, "near_singular": near_singular,
                "step_used": step}

        if pos_err > self.accept_pos or rot_err > self.accept_rot:
            # 残差大到不像「够不到」，像目标本身有问题。这一帧丢掉。
            self.stats["rejected"] += 1
            info["reason"] = (f"residual too large: {pos_err*1000:.0f}mm "
                              f"{np.rad2deg(rot_err):.0f}° — 目标可能是脏的")
            return None, info

        self.stats["ok"] += 1
        if pos_err > pos_tol or rot_err > rot_tol:
            # 没解到目标上，但交出的是**够得到的里面最近的**。臂会顶在边界上等。
            self.stats["slipped"] += 1
            info["reason"] = (f"best-effort: {pos_err*1000:.1f}mm "
                              f"{np.rad2deg(rot_err):.1f}°"
                              + (f" [近奇异减速 w={manip:.4f}<{self.min_manip}]"
                                 if near_singular else
                                 f" [跟不上，行程已用 {b_use*100:.0f}%]" if stuck else ""))
        else:
            info["reason"] = "converged"
        return q, info

    def report(self) -> str:
        s = self.stats
        n = max(1, s["calls"])
        return (f"OptIK  调用 {s['calls']}  给解 {s['ok']} ({100*s['ok']/n:.1f}%)  "
                f"其中打滑 {s['slipped']}  丢帧 {s['rejected']}  "
                f"平均 {s['nfev']/n:.1f} 次代价求值  "
                f"平均 {1000*s['seconds']/n:.2f} ms/次")


class ToolFrameKinematics:
    _overrides_fk = True        # 告诉 AR5OptIK：别抄 _chain 的近路
    """把运动学从**法兰系**搬到**工具系**。包在任何一个 kinematics/求解器外面。

    为什么需要：`--tcp-z` 只通过 `setEndEffectorFrame` 告诉了**控制器**
    （给力控和碰撞检测用），本地这套运动学根本不知道。于是
    `fk(q)` 算的是法兰位姿、`ik(T)` 也把 T 当成法兰位姿 —— 遥操全程在法兰系里跑。

    后果不是差一个常数偏移那么简单：**旋转中心错了**。
    你手腕原地转，法兰不动，而真正的灵巧手在 150mm 之外画弧：

        转 15° → 手被甩出  39mm
        转 30° → 手被甩出  78mm
        转 45° → 手被甩出 115mm
        转 90° → 手被甩出 212mm

    操作者的感受就是「我只是转了个手腕，机械手却跑掉了」。

    换算很简单，T_tool = T_flange · X（X 是工具相对法兰的固定变换）：
        fk_tool(q)   = fk_flange(q) · X
        ik_tool(T)   = ik_flange(T · X⁻¹)
    雅可比也要换：工具点的线速度多了 ω × r 那一项。
    """

    def __init__(self, base, tcp_offset):
        self._base = base
        off = np.asarray(tcp_offset, dtype=float).reshape(-1)
        X = np.eye(4)
        X[:3, 3] = off[:3]
        if off.shape[0] >= 6 and np.linalg.norm(off[3:6]) > 1e-12:
            v = off[3:6]
            th = float(np.linalg.norm(v))
            k = v / th
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            X[:3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        self.X = X
        self.X_inv = np.linalg.inv(X)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def fk(self, q):
        return self._base.fk(q) @ self.X

    def jacobian(self, q):
        """法兰雅可比 → 工具雅可比。角速度不变，线速度加上 ω × r。"""
        J = self._base.jacobian(q)
        R = self._base.fk(q)[:3, :3]
        r = R @ self.X[:3, 3]                     # 法兰→工具，在基座系里
        Jt = J.copy()
        Jt[:3, :] += np.cross(J[3:, :].T, r).T    # v_tool = v_flange + ω × r
        return Jt

    def ik(self, T_target, q_seed, **kw):
        return self._base.ik(np.asarray(T_target, dtype=float) @ self.X_inv,
                             q_seed, **kw)
