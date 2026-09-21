#!/usr/bin/env python3
"""Vision Pro → AR5 遥操，servoJ 版（覆盖式关节目标流，控制器插补）。

和 MoveL 版（avp_movel_teleop.py）的本质区别：
  MoveL  每个点是必须走完的任务，执行时间 ≈ 2√(L/a)+74ms，段长随手速增长
         → 实测延迟 240~500ms 且手越快越差。
  servoJ 每 T 秒给一个**关节角目标**，控制器按前瞻时间插补到 1ms，
         新目标直接覆盖旧的，没有队列、没有任务开销。
         实测（probe_servoj.py）：T=8ms 滞后 24ms，T=20ms 滞后 60ms，
         幅值保持 ~100%，1Hz 正弦跟得住，滞后不随频率变化。

链路（每 T 秒一次）：
    头显手腕位姿 → 跳变门 → 卡尔曼 → One-Euro → 绝对 TCP 目标 → 工作空间钳位
      → **本地逆解**（旋量法解析解 ~2ms） → 关节限位 + 单帧步长限幅
      → sendCommand(JointPosition)

安全设计：
  * 单帧关节步长上限 --max-step-rad，超了就按比例缩，臂永远不会跳。
  * 逆解失败/超限/头显断流 → 保持当前关节角继续发（不停流，停流控制器会报通信丢包）。
  * 摘离合 → 保持当前位置，重新捏合时就地重新锚定。
  * q 退出或异常 → 发一条 finished、stopServoJoint、切回 NRT。

按 ENTER 挂/摘离合，右手保持捏合驱动臂；q 退出；h 停下并关节回位。
"""
from __future__ import annotations

import argparse
import datetime
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
AR5_ROOT = "/home/crp-5070ti-01/yuhang_workspace/vel_ar5/openvla-energy"
sys.path.insert(0, AR5_ROOT)

from avp_arm_teleop import (OneEuro, filter_rotation, yaw_matrix, mirror_matrix,  # noqa: E402
                            wrist_xyz, wrist_R, matrix_to_axis_angle, axis_angle_to_matrix,
                            connect_streamer, HandTracker, PINCH_ON, PINCH_OFF, DEFAULT_CALIB)
from avp_movel_teleop import JumpGate, KalmanCV, RunLog, e2e_report  # noqa: E402
from hand_retarget import Calibration, Retargeter  # noqa: E402
from experiments.robot.ar5.ar5_env import AR5Env  # noqa: E402
from experiments.robot.ar5.config import AR5Config  # noqa: E402
import inspire_hand6 as ih  # noqa: E402


class TrapTracker:
    """每关节独立的梯形速度规划：限速 + 限加速度，朝**最新**目标平滑逼近。

    每周期做三件事：
      1. 按刹车律定这一步该多快：v_des = sign(e)·min(V, √(2·A_b·|e|), |e|/dt)
         第二项保证从当前速度全力减速能正好停在目标上（不冲过头），
         第三项保证最后一步不越过目标。
      2. 实际速度朝 v_des 靠，每周期最多变 A·dt —— 这就是加速度限制，
         也是梯形和「只限步长」的区别：只限步长时速度可以一帧从 0 跳到上限。
      3. q += v·dt

    刹车用 A_b = brake·A（默认 0.7A）而不是满 A：离散步进下 v 跟不上 v_des 的
    下降沿，留点余量才不会到点还有速度、被限位一把拍死（那是无穷大减速度）。

    目标随时可以变（遥操每周期都在变），它不规划整条轨迹，只做当前这一步。
    """

    def __init__(self, q0, dt, v_max, a_max, brake=0.7):
        self.q = np.asarray(q0, dtype=float).copy()
        self.v = np.zeros_like(self.q)
        self.dt, self.v_max, self.a_max, self.brake = dt, v_max, a_max, brake
        self.n_vel_sat = self.n_acc_sat = 0

    def reset(self, q):
        self.q = np.asarray(q, dtype=float).copy()
        self.v[:] = 0.0

    def step(self, q_goal, lo, hi):
        dt, V, A = self.dt, self.v_max, self.a_max
        q_goal = np.clip(np.where(np.isfinite(q_goal), q_goal, self.q), lo, hi)
        e = q_goal - self.q
        v_brake = np.sqrt(2.0 * self.brake * A * np.abs(e))
        v_des = np.sign(e) * np.minimum.reduce([np.full_like(e, V), v_brake, np.abs(e) / dt])
        self.n_vel_sat += int(np.any(np.abs(v_des) >= V - 1e-9))
        dv = np.clip(v_des - self.v, -A * dt, A * dt)
        self.n_acc_sat += int(np.any(np.abs(v_des - self.v) > A * dt + 1e-12))
        self.v += dv
        self.q += self.v * dt
        # 限位兜底：撞上就把该轴速度清零（这一步是无穷大减速度，只作最后一道网）
        out = (self.q < lo) | (self.q > hi)
        if np.any(out):
            self.q = np.clip(self.q, lo, hi)
            self.v[out] = 0.0
        return self.q


class DelayedSpline:
    """延迟缓冲 + 穿过多点的样条。**牺牲固定延迟换顺滑**。

    两点插值只有「当前点」和「刚解出的目标」，终点速度只能由过去两点外推 —— 目标一抖
    速度估计就抖，样条跟着过冲。这里改成：把逆解结果连时间戳存进缓冲区，输出的是
    **lag 毫秒之前**那一刻的值。因为落后了，输出点的两侧都有真实数据，可以做真正的
    内插而不是外推，切线用中心差分，天然 C1 连续、不过冲。

    这就是桌面 demo.cpp（MoveIt 迭代样条参数化 + 重采样）的做法：先有一段路径，再
    打时间戳、按固定 dt 重采样。区别只是我们的「路径」是滚动的最近几个点。

    代价就是那个 lag，实打实加在端到端延迟上。lag 至少要覆盖两个目标间隔
    （头显 ~16.7ms → 至少 35ms），否则右边没有足够的点，退化回外推。
    """

    def __init__(self, q0, dt, lag, v_max, a_max):
        self.dt, self.lag = float(dt), float(lag)
        self.v_max, self.a_max = float(v_max), float(a_max)
        self.q = np.asarray(q0, dtype=float).copy()
        self.v = np.zeros_like(self.q)
        self.a = np.zeros_like(self.q)
        self.buf = []                      # [(t, q)]，按时间升序
        self.n_vel_sat = self.n_acc_sat = self.n_extrap = 0

    def reset(self, q):
        self.q = np.asarray(q, dtype=float).copy()
        self.v[:] = 0.0
        self.a[:] = 0.0
        self.buf = []

    def push(self, q_goal, t, lo=None, hi=None):
        """收到新的逆解结果时调用。"""
        q_goal = np.asarray(q_goal, dtype=float)
        if lo is not None:
            q_goal = np.clip(q_goal, lo, hi)
        if self.buf and t <= self.buf[-1][0]:
            t = self.buf[-1][0] + 1e-4
        self.buf.append((float(t), q_goal.copy()))
        # 只留够用的历史：lag 两侧各几个点
        keep = self.lag + 0.30
        while len(self.buf) > 4 and t - self.buf[0][0] > keep:
            self.buf.pop(0)

    def step(self, t_now, lo=None, hi=None):
        """每个控制周期调用：返回 t_now − lag 那一刻的关节角。"""
        if not self.buf:
            return self.q
        t_out = t_now - self.lag
        B = self.buf
        if t_out <= B[0][0]:
            q_new = B[0][1].copy()
        elif t_out >= B[-1][0]:
            q_new = B[-1][1].copy()        # 右边没数据了：保持，不外推
            self.n_extrap += 1
        else:
            i = 0
            for k in range(len(B) - 1):
                if B[k][0] <= t_out < B[k + 1][0]:
                    i = k
                    break
            t0, q0 = B[i]
            t1, q1 = B[i + 1]
            h = max(t1 - t0, 1e-6)
            s = (t_out - t0) / h
            # 切线用中心差分（Catmull-Rom）：左右都有真实点，不用外推
            tm, qm = B[i - 1] if i - 1 >= 0 else (t0 - h, q0)
            tp, qp = B[i + 2] if i + 2 < len(B) else (t1 + h, q1)
            m0 = (q1 - qm) / max(t1 - tm, 1e-6)
            m1 = (qp - q0) / max(tp - t0, 1e-6)
            m0 = np.clip(m0, -self.v_max, self.v_max)
            m1 = np.clip(m1, -self.v_max, self.v_max)
            s2, s3 = s * s, s * s * s
            h00, h10, h01, h11 = 2*s3-3*s2+1, s3-2*s2+s, -2*s3+3*s2, s3-s2
            q_new = h00*q0 + h10*h*m0 + h01*q1 + h11*h*m1
            d00, d01 = 6*s2-6*s, -6*s2+6*s
            d10, d11 = 3*s2-4*s+1, 3*s2-2*s
            self.v = (d00*q0 + d01*q1)/h + d10*m0 + d11*m1
            e00, e01, e10, e11 = 12*s-6, -12*s+6, 6*s-4, 6*s-2
            self.a = (e00*q0 + e01*q1)/(h*h) + (e10*m0 + e11*m1)/h
            self.n_vel_sat += int(np.any(np.abs(self.v) > 0.98 * self.v_max))
            self.n_acc_sat += int(np.any(np.abs(self.a) > 0.98 * self.a_max))
        if lo is not None:
            q_new = np.clip(q_new, lo, hi)
        self.q = q_new
        return self.q


class RuckigTracker:
    """Ruckig 在线轨迹生成：限速 + 限加速度 + **限加加速度**的 S 曲线。

    和上面的梯形比，多了一条 jerk 限制：梯形的加速度可以一周期从 0 跳到 A，
    那是无穷大 jerk，电机听到的是方波，就是「嗡」的来源。Ruckig 把加速度也
    做成斜坡，同时保证时间最优。每周期从**当前状态**朝最新目标重新规划，
    目标随时可变，正是遥操的工况。

    还接受目标速度（由相邻两次逆解解出的关节目标差分估计）：跟一个匀速移动
    的目标时，只靠位置误差必然有稳态滞后，把目标速度一起给出去就没有这个滞后。
    """

    def __init__(self, q0, dt, v_max, a_max, j_max, feedforward=True):
        from ruckig import Ruckig, InputParameter, OutputParameter, ControlInterface
        self.n = len(q0)
        self.dt = dt
        self.otg = Ruckig(self.n, dt)
        self.inp = InputParameter(self.n)
        self.out = OutputParameter(self.n)
        self.CI = ControlInterface
        self.inp.max_velocity = [float(v_max)] * self.n
        self.inp.max_acceleration = [float(a_max)] * self.n
        self.inp.max_jerk = [float(j_max)] * self.n
        self.q = np.asarray(q0, dtype=float).copy()
        self.v = np.zeros(self.n)
        self.a = np.zeros(self.n)
        self.feedforward = feedforward
        self._prev_goal = None
        self._since_goal = 0.0
        self._last_v_goal = np.zeros(self.n)
        self.n_vel_sat = self.n_acc_sat = self.n_err = 0

    def reset(self, q):
        self.q = np.asarray(q, dtype=float).copy()
        self.v[:] = 0.0
        self.a[:] = 0.0
        self._prev_goal = None

    def step(self, q_goal, lo, hi):
        q_goal = np.clip(np.where(np.isfinite(q_goal), q_goal, self.q), lo, hi)
        v_goal = np.zeros(self.n)
        if self.feedforward and self._prev_goal is not None:
            # 只有目标真的变了才更新前馈：目标以 60Hz 更新而这里 125Hz 调用，
            # 按 self.dt 算会把速度高估到 3 倍（2026-09-21 离线复现）
            moved = float(np.max(np.abs(q_goal - self._prev_goal)))
            self._since_goal += self.dt
            if moved > 1e-9:
                v_goal = np.clip((q_goal - self._prev_goal) / max(self._since_goal, self.dt),
                                 -np.asarray(self.inp.max_velocity), np.asarray(self.inp.max_velocity))
                self._prev_goal = q_goal.copy()
                self._last_v_goal = v_goal.copy()
                self._since_goal = 0.0
            else:
                v_goal = self._last_v_goal          # 目标没变：沿用上次估的速度
        elif self._prev_goal is None:
            self._prev_goal = q_goal.copy()
        inp, out = self.inp, self.out
        inp.control_interface = self.CI.Position
        inp.current_position = self.q.tolist()
        inp.current_velocity = self.v.tolist()
        inp.current_acceleration = self.a.tolist()
        inp.target_position = q_goal.tolist()
        inp.target_velocity = v_goal.tolist()
        inp.target_acceleration = [0.0] * self.n
        res = self.otg.update(inp, out)
        if int(res) < 0:                       # 输入非法（状态超限等）：保持，下一周期重来
            self.n_err += 1
            return self.q
        self.q = np.asarray(out.new_position, dtype=float)
        self.v = np.asarray(out.new_velocity, dtype=float)
        self.a = np.asarray(out.new_acceleration, dtype=float)
        vm, am = np.asarray(inp.max_velocity), np.asarray(inp.max_acceleration)
        self.n_vel_sat += int(np.any(np.abs(self.v) > 0.98 * vm))
        self.n_acc_sat += int(np.any(np.abs(self.a) > 0.98 * am))
        out_of = (self.q < lo) | (self.q > hi)
        if np.any(out_of):
            self.q = np.clip(self.q, lo, hi)
            self.v[out_of] = 0.0
            self.a[out_of] = 0.0
        return self.q


class CubicInterp:
    """两点三次样条插值。参考 MoveIt 的 IterativeSplineParameterization + 样条重采样
    （桌面/路径点时间参数化/demo.cpp）：先按速度/加速度上限给两点之间定时长，
    再按固定 dt 重采样，每个控制周期取一个点发出。

    和那份 demo 的区别：demo 是离线给一整条路径打时间戳再一次性重采样；遥操只有
    「上一个已发点」和「刚解出的新目标」两个点，而且目标每帧都在变，所以是
    **每来一个新目标就用当前状态重新拟合一段**，不排队、不累积。

    三次 Hermite：q(s) = h00·q0 + h10·T·v0 + h01·q1 + h11·T·v1,  s = t/T
      h00 = 2s³−3s²+1   h10 = s³−2s²+s   h01 = −2s³+3s²   h11 = s³−s²
    段时长 T 由两条约束定（对每个关节取最紧的）：
      速度：三次段的峰值速度 ≈ 1.5·|Δq|/T          → T ≥ 1.5·|Δq|/v_max
      加速度：端点加速度 |6Δq − T(4v0+2v1)|/T²     → 解出 T 的下界
    取满足全部关节的最大值，再夹到 ≥ dt（至少一个控制周期）。
    """

    def __init__(self, q0, dt, v_max, a_max, feedforward=True, v_tau=0.04):
        self.v_tau = float(v_tau)
        self._v_filt = np.zeros_like(np.asarray(q0, dtype=float))
        self.q = np.asarray(q0, dtype=float).copy()
        self.v = np.zeros_like(self.q)
        self.a = np.zeros_like(self.q)
        self.dt, self.v_max, self.a_max = dt, float(v_max), float(a_max)
        self.feedforward = feedforward
        self._prev_goal = None
        self.seg_T = dt              # 当前段总时长
        self.seg_t = 0.0             # 段内已走时间
        self._prev_dt = dt           # 上一次目标间隔
        self._prev_vel = np.zeros_like(self.q)   # 上一段终点速度，保证 C1 连续
        self._seg = None             # (q0,v0,q1,v1,T)
        self.n_vel_sat = self.n_acc_sat = self.n_err = 0
        self.seg_times = []          # 统计用：每段时长

    def reset(self, q):
        self.q = np.asarray(q, dtype=float).copy()
        self.v[:] = 0.0
        self.a[:] = 0.0
        self._prev_goal = None
        self._prev_vel = np.zeros_like(self.q)
        self._v_filt = np.zeros_like(self.q)
        self._seg = None

    def _segment_time(self, q0, v0, q1, v1):
        """满足速度和加速度上限的最短段时长。"""
        d = np.abs(q1 - q0)
        T = self.dt
        # 速度：峰值 ≈ 1.5·Δq/T（两端速度为 0 的三次段）
        with np.errstate(divide="ignore", invalid="ignore"):
            T = max(T, float(np.max(1.5 * d / self.v_max)) if self.v_max > 0 else T)
        # 加速度：|6Δq − T(4v0+2v1)|/T² ≤ a_max，迭代两次（T 出现在两边）
        for _ in range(3):
            num = np.abs(6.0 * (q1 - q0) - T * (4.0 * v0 + 2.0 * v1))
            num = np.maximum(num, np.abs(-6.0 * (q1 - q0) + T * (2.0 * v0 + 4.0 * v1)))
            need = np.sqrt(np.maximum(num, 0.0) / max(self.a_max, 1e-9))
            T = max(self.dt, float(np.max(need)), T * 0.999)
        return float(T)

    def set_goal(self, q_goal, dt_goal, lo, hi):
        """**只在拿到新的逆解结果时调用**（头显新帧，约每 16.7ms 一次）。

        段时长取两次逆解之间的真实间隔 dt_goal，这样这一段正好在下一个目标到来时走完，
        插值只是把 16.7ms 填成 dt 的网格。只有当这个时长顶不住速度/加速度上限时才拉长。

        ⚠ 段时长不能按限制反推：那样算出来 149ms，而每周期只走 8ms 就重新拟合，
        永远停在段的起始慢速段，实测滞后 141ms（2026-09-21 离线对照）。
        """
        q_goal = np.clip(np.where(np.isfinite(q_goal), q_goal, self.q), lo, hi)
        # 段起点 = **当前实际指令位置和速度**，不是上一个目标：段没走完就换目标时，
        # 从上一个目标起算会产生跳变，被单帧步长限幅拦掉（2026-09-21 实测 63% 的
        # 周期都在限幅）。从当前状态起算天然 C1 连续。
        q0, v0 = self.q.copy(), (self.v.copy() if self.feedforward else np.zeros_like(q_goal))
        T = max(self.dt, float(dt_goal))
        # 终点速度 = 目标自己的移动速度，用**真实目标间隔**算。
        # 不能用被拉长的 T：T 一拉长 v1 就变小，下一段因此需要更长的 T，
        # 正反馈直到臂几乎不动（2026-09-21 离线复现）。
        v1 = np.zeros_like(q_goal)
        if self.feedforward and self._prev_goal is not None:
            raw_v = np.clip((q_goal - self._prev_goal) / max(dt_goal, self.dt),
                            -self.v_max, self.v_max)
            # 目标速度是两个逆解结果的差分，目标有噪声它就抖，样条跟着过冲回摆 —— 这是
            # servoJ 下残余抖动的主要来源。一阶低通，时间常数 v_tau。
            if self.v_tau > 0:
                a = dt_goal / (dt_goal + self.v_tau)
                self._v_filt = (1 - a) * self._v_filt + a * raw_v
                v1 = self._v_filt.copy()
            else:
                v1 = raw_v
        # 只按**速度**上限拉长段时长。按加速度拉长会触发上面那个正反馈，
        # 而加速度实际由目标本身的平滑度决定（卡尔曼 + One-Euro 已经滤过）。
        d = np.max(np.abs(q_goal - q0))
        if self.v_max > 0 and d / T > self.v_max:
            T = d / self.v_max
        self._seg = (q0.copy(), v0.copy(), q_goal.copy(), v1.copy(), T)
        self._prev_goal, self._prev_dt, self._prev_vel = q_goal.copy(), dt_goal, v1.copy()
        self.seg_T, self.seg_t = T, 0.0
        self.seg_times.append(T)

    def step(self, q_goal=None, lo=None, hi=None):
        """一个控制周期：沿当前段前进 dt。不重新拟合。"""
        if self._seg is None:
            return self.q
        q0, v0, q1, v1, T = self._seg
        self.seg_t = min(self.seg_t + self.dt, T)
        s = self.seg_t / T
        s2, s3 = s * s, s * s * s
        h00, h10, h01, h11 = 2*s3-3*s2+1, s3-2*s2+s, -2*s3+3*s2, s3-s2
        q = h00*q0 + h10*T*v0 + h01*q1 + h11*T*v1
        # 一阶、二阶导（对 t）
        d00, d10, d01, d11 = 6*s2-6*s, 3*s2-4*s+1, -6*s2+6*s, 3*s2-2*s
        v = (d00*q0 + d01*q1)/T + d10*v0 + d11*v1
        e00, e10, e01, e11 = 12*s-6, 6*s-4, -12*s+6, 6*s-2
        a = (e00*q0 + e01*q1)/(T*T) + (e10*v0 + e11*v1)/T
        self.n_vel_sat += int(np.any(np.abs(v) > 0.98 * self.v_max))
        self.n_acc_sat += int(np.any(np.abs(a) > 0.98 * self.a_max))
        if lo is not None:
            out = (q < lo) | (q > hi)
            q = np.clip(q, lo, hi)
            if np.any(out):
                v = v.copy(); a = a.copy()
                v[out] = 0.0
                a[out] = 0.0
        self.q, self.v, self.a = q, v, a
        return self.q


class ServoJSession:
    """servoJ 会话：进入切 RT 并开启 servoJ，退出一定还原 NRT。"""

    def __init__(self, env, period, lookahead, kp, log=print):
        self.env, self.log = env, log
        self.r, self.sdk = env.arm._robot, env.arm.sdk
        self.period, self.lookahead, self.kp = period, lookahead, kp
        self.n = int(env.config.arm.num_joints)
        self.rt = None
        self.streaming = self.servo_on = self.moving = False
        self.sent = self.late = 0

    def call(self, fn, *a):
        ec = {}
        out = fn(*a, ec)
        if ec.get("ec", 0):
            raise RuntimeError(f"{getattr(fn,'__name__','sdk')}: {ec}")
        return out

    def joints(self):
        # jointPos 返回「机器人本体 + 外部轴」，这台是 13 个；前 7 个才是臂。
        return np.asarray(self.call(self.r.jointPos), dtype=float)[:self.n]

    def _power_on(self):
        s, r = self.sdk, self.r
        for attempt in range(6):
            t_idle = time.monotonic()
            while time.monotonic() - t_idle < 3.0:
                if self.call(r.operationState) == s.OperationState.idle:
                    break
                time.sleep(0.05)
            self.call(r.setOperateMode, s.OperateMode.automatic)
            self.call(r.setPowerState, True)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 2.0:
                if str(self.call(r.powerState)).endswith(".on"):
                    return
                time.sleep(0.1)
            self.log(f"  [servoJ] 上电未成功（第 {attempt+1} 次），1s 后重试")
            time.sleep(1.0)
        raise RuntimeError(f"电机上不了电: {self.call(r.powerState)}")

    def __enter__(self):
        s, r = self.sdk, self.r
        self.call(r.setMotionControlMode, s.MotionControlMode.NrtCommandMode)
        self._power_on()
        # 实时状态流：不订阅的话 jointPos 返回不刷新的快照（驱动里踩过）
        r.startReceiveRobotState(datetime.timedelta(milliseconds=1), ["q_m"])
        self.streaming = True
        self.call(r.setMotionControlMode, s.MotionControlMode.RtCommandMode)
        self.rt = r.getRtMotionController()
        ec = {}
        self.rt.setServoJoint(float(self.period), float(self.lookahead), float(self.kp), ec)
        if ec.get("ec", 0):
            raise RuntimeError(f"setServoJoint: {ec}")
        self.servo_on = True
        self.rt.startMove(s.RtControllerMode.jointPosition)
        self.moving = True
        self.log(f"  [servoJ] T={self.period*1000:.0f}ms 前瞻={self.lookahead*1000:.0f}ms Kp={self.kp}")
        return self

    def send(self, q):
        self.rt.sendCommand(self.sdk.JointPosition([float(v) for v in q]))
        self.sent += 1

    def __exit__(self, *exc):
        s, r = self.sdk, self.r
        try:
            if self.moving:
                try:
                    cmd = s.JointPosition([float(v) for v in self.joints()])
                    cmd.setFinished()
                    self.rt.sendCommand(cmd)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self.rt.stopMove()
                except Exception:  # noqa: BLE001
                    pass
            if self.servo_on:
                try:
                    self.rt.stopServoJoint()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            if self.streaming:
                try:
                    r.stopReceiveRobotState()
                except Exception:  # noqa: BLE001
                    pass
            try:
                ec = {}
                r.setMotionControlMode(s.MotionControlMode.NrtCommandMode, ec)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
            self.log(f"  [servoJ] 已关闭，发送 {self.sent} 条，迟到 {self.late} 条")
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Vision Pro → AR5 遥操（servoJ，覆盖式）",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("ip", help="Vision Pro 的 IP")
    # servoJ
    p.add_argument("--period", type=float, default=0.008,
                   help="servoJ 发送周期 s。**滞后≈3×周期**（实测 8ms→24ms, 20ms→60ms, 33ms→99ms）。"
                        "范围 (0.001,0.1]，太小会伺服报错，太大会判通信丢包")
    p.add_argument("--lookahead", type=float, default=0.008,
                   help="前瞻时间 s。0=按实际到达间隔算速度（要求发送很准时）；"
                        "非 0=按固定周期算，能抹平抖动。一般取周期或 2~3 倍")
    p.add_argument("--kp", type=float, default=1.0, help="位置反馈增益，0=纯前馈。振荡就调小，滞后大就调大")
    p.add_argument("--max-vel", type=float, default=1.2,
                   help="梯形规划的关节速度上限 rad/s（1.2rad/s ≈ 69°/s）。越大越跟手，越容易过冲")
    p.add_argument("--max-acc", type=float, default=8.0,
                   help="梯形规划的关节加速度上限 rad/s²。**这是顺滑度的主旋钮**："
                        "只限速度时目标一跳速度就从 0 到上限，等于无穷大加速度，那就是顿挫的来源")
    p.add_argument("--planner", choices=("cubic", "spline", "ruckig", "trap", "none"), default="cubic",
                   help="cubic=两点三次样条插值（默认，外推）；"
                        "**spline=延迟缓冲+穿过多点的样条**（内插，最顺滑，代价是固定延迟 --lag-ms）；"
                        "ruckig=限 jerk 的 S 曲线在线规划；trap=梯形；"
                        "none=第一版行为（逆解直接发，只有步长限幅），做 A/B 用")
    p.add_argument("--lag-ms", type=float, default=50.0,
                   help="spline 规划器的延迟缓冲 ms。**用它换顺滑**：落后这么多就能拿到输出点两侧"
                        "的真实数据做内插而不是外推。至少要 35ms（覆盖两个头显帧间隔），"
                        "小于这个值右边没点会退化回保持。50~80 顺滑，30 跟手")
    p.add_argument("--max-jerk", type=float, default=200.0,
                   help="Ruckig 的关节加加速度上限 rad/s³。越小越顺但起停越肉；"
                        "到达满加速度需要 max-acc/max-jerk 秒")
    p.add_argument("--predict", action=argparse.BooleanOptionalAction, default=True,
                   help="帧间预测：用卡尔曼估的手速把头显位置外推到当前控制周期。"
                        "头显 60Hz 而控制周期 8ms，不外推的话中间两周期目标是陈旧的，"
                        "规划器的速度前馈会按错误的间隔算")
    p.add_argument("--predict-max", type=float, default=0.03,
                   help="帧间预测最多外推多少秒。头显丢帧时防止外推跑飞")
    p.add_argument("--v-tau", type=float, default=0.0,
                   help="插值终点速度的低通时间常数 s。⚠ 离线实测**调大反而更抖**"
                        "（滤后的速度和目标实际速度对不上，样条在段间来回修正），默认 0=不平滑。"
                        "抖动要从 --lp-cutoff / --w-psi 那边治")
    p.add_argument("--no-ff", action="store_true",
                   help="关掉目标速度前馈。前馈用相邻两次逆解的差分估计目标速度，"
                        "跟匀速目标时消除稳态滞后；目标噪声大时可以关掉")
    p.add_argument("--brake", type=float, default=0.7,
                   help="刹车时用多少比例的加速度。1.0=满力刹车，离散步进下会到点还有速度；0.5~0.8 合适")
    p.add_argument("--max-step-rad", type=float, default=0.03,
                   help="单帧每关节变化量的硬上限 rad，梯形规划之外的最后一道网。8ms 下 0.03rad=3.75rad/s")
    # 滤波
    p.add_argument("--kf-sigma-a", type=float, default=5.0, help="卡尔曼过程噪声 m/s²")
    p.add_argument("--kf-sigma-m", type=float, default=0.002, help="卡尔曼量测噪声 m")
    p.add_argument("--lp-cutoff", type=float, default=3.0, help="One-Euro 最小截止频率 Hz，0=关")
    p.add_argument("--lp-beta", type=float, default=2.0, help="One-Euro 速度系数")
    p.add_argument("--hand-speed-max", type=float, default=0.0, help="跳变门 m/s，0=关")
    p.add_argument("--target-cutoff", type=float, default=4.0,
                   help="**进逆解前**对目标位姿再滤一级的 One-Euro 截止频率 Hz。"
                        "前面的滤波在头显原始位姿上，之后帧间预测（手速×时间）和坐标映射"
                        "会重新引入噪声，这一级直接滤送进逆解的目标。抖就往小调（2~3），0=关")
    p.add_argument("--target-beta", type=float, default=1.0,
                   help="上面那级滤波的速度系数。大=快动作更跟手，小=更稳")
    # 映射
    p.add_argument("--yaw", type=float, default=-90.0)
    p.add_argument("--mirror", choices=("none", "lr", "fb"), default="none")
    p.add_argument("--scale", type=float, default=0.4, help="手动 1m 末端动 scale m")
    p.add_argument("--rot-mode", choices=("none", "roll", "full"), default="roll")
    p.add_argument("--rot-scale", type=float, default=1.0)
    # 2026-09-21：60/20 -> 20/10。盒子本身已经按臂的实际可达范围重新量过并放大，
    # 再往里收 60mm 就把刚放出来的量又收回去了。够不到的目标由逆解拒掉，不靠这道留边。
    p.add_argument("--ws-margin-mm", type=float, default=20.0, help="工作空间盒 x/y 向内收")
    p.add_argument("--ws-margin-z-mm", type=float, default=10.0, help="工作空间盒 z 向内收")
    p.add_argument("--max-stale", type=float, default=0.25)
    # 逆解
    p.add_argument("--ik", choices=("geo", "poe", "driver"), default="geo",
                   help="geo=闭式解+选解层（臂角锚定，肘不游走，~3.3ms，推荐）；"
                        "poe=只有闭式解（臂角跟上一帧漂，画个圆漂 30°，肘乱动）；driver=驱动数值解")
    p.add_argument("--w-limit", type=float, default=4.0,
                   help="选解代价里「关节贴限位」的权重。大=宁可位形变一点也要远离限位")
    p.add_argument("--w-psi", type=float, default=0.15,
                   help="选解代价里「臂角偏离锚定值」的权重。大=肘更钉死，小=更跟手")
    p.add_argument("--ik-max-jump", type=float, default=0.25,
                   help="逆解候选相对上一帧的最大关节跳变 rad。同构型里也有差半个臂角的远解，"
                        "限位代价可能盖过连续性，实测出现过 1.55rad 跳变；这道门是硬的")
    p.add_argument("--psi-span", type=float, default=0.5,
                   help="臂角搜索窗口半宽 rad。小=肘更稳，大=够得到的位姿更多")
    # 臂 / 手
    p.add_argument("--robot-ip", default="192.168.2.160")
    p.add_argument("--local-ip", default="192.168.2.222")
    p.add_argument("--no-home", action="store_true")
    p.add_argument("--no-hand", action="store_true")
    p.add_argument("--hand-port", default="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0")
    p.add_argument("--hand-hz", type=float, default=40)
    p.add_argument("--hand-force", type=int, default=1000,
                   help="力控阈值，范围 0~1000，**1000 是满档**。手内部固件执行：手指顶到"
                        "这个力才停。调小是防夹坏东西和堵转，调大是抓得住")
    p.add_argument("--hand-speed", type=int, default=1000)
    p.add_argument("--smooth", type=float, default=0.20)
    p.add_argument("--pinch-gate", type=float, default=0.45)
    p.add_argument("--freeze-thumb-rot", action="store_true")
    p.add_argument("--calib-file", default=str(DEFAULT_CALIB))
    p.add_argument("--print-hz", type=float, default=5)
    p.add_argument("--log", default="auto", help="运行日志路径；none=不写")
    # ── 数采 ──
    p.add_argument("--record", action="store_true",
                   help="开启数采。空格开录，s 存并标成功，← 存但不标成功，退格丢弃。不开则不碰相机")
    p.add_argument("--out-dir", default="~/atom_episodes", help="数据存哪，每条一个 npz + 每路相机一个 mp4")
    p.add_argument("--task", default="", help="这条数据在做什么，一句英文小写祈使句，会写进每个 npz")
    p.add_argument("--record-hz", type=float, default=30.0,
                   help="录制帧率。相机是 30fps，高过它只会录到重复帧")
    p.add_argument("--num-episodes", type=int, default=0,
                   help="这一次采多少条就收工。0=不限")
    p.add_argument("--frame-cap", type=int, default=0,
                   help="一条最多多少帧，到了自动存盘。**按帧数不按秒**，回路周期抖一下"
                        "不该改变一条的长短。30Hz 下 900 帧 = 30 秒。0=不限")
    p.add_argument("--reset-seconds", type=float, default=5.0,
                   help="存完一条之后的复位时间。这段时间里空格不响应，防止现场还没摆好就开录。0=不等")
    p.add_argument("--auto-record", action="store_true",
                   help="接合即开录、摘离合即存盘，不用按键。**一般别用**：现场还没摆好就开录，"
                        "数据看着正常但前几秒是空的")
    p.add_argument("--cam-top", default="",
                   help="顶部相机设备路径，留空用 config 里的。**必须用 /dev/v4l/by-path/**，"
                        "两台相机型号和序列号都一样，by-id 只有一条，video0/2 拔插会互换")
    p.add_argument("--cam-front", "--cam-wrist", dest="cam_front", default="",
                   help="第二路相机设备路径，同上。**目前这台不在腕上**，它固定在台架上"
                        "侧对着臂，所以数据里的键名是 cam_front。真装到腕上了再改名")
    return p.parse_args()


def setup_hand(args, calib):
    """和 avp_arm_teleop.py 里验证过的那段一致，别自己另写一套。"""
    if args.no_hand:
        return None
    from inspire_hand6 import INDEX as _IDX, THUMB_BEND as _TB
    hand = ih.InspireHand6(port=args.hand_port)
    hand.connect()
    # 钳位取标定文件和驱动默认值的交集；食指/拇指弯曲除外，那两维走驱动的
    # 耦合下限，老标定里的静态下限会把耦合作废（表现为「单独弯食指只能走一半」）
    for i in range(ih.NUM_DOF):
        if i in (_IDX, _TB):
            calib.limits[i] = hand.limits[i] = (0, 1000)
            continue
        clo, chi = calib.limits.get(i, hand.limits[i])
        hlo, hhi = hand.limits[i]
        hand.limits[i] = (max(clo, hlo), min(chi, hhi))
    ok, why = hand.liveness()
    if not ok:
        print("\n" + "!" * 62)
        print("!! 灵巧手的控制循环没在跑 —— 跟踪也没用，手指不会动。")
        print(f"!!   {why}")
        print("!! → **给手断电重启**（手是独立 24V 供电，不是 USB 供的），等 5 秒再上。")
        print("!" * 62 + "\n")
        raise RuntimeError(f"灵巧手未就绪: {why}")
    hand.set_force(args.hand_force)
    hand.set_speed(args.hand_speed)
    hand.start_writer()
    print(f"灵巧手已连接 {args.hand_port}  力控 {args.hand_force}g  速度 {args.hand_speed}")
    return hand


def _end_episode(rec, index, task, success=True):
    """存下当前这条，返回下一条的编号。存盘在后台线程，主循环不停。"""
    if len(rec) == 0:
        print("\n  一帧都没录到，跳过")
        return index
    notes = rec.warnings()
    n, sec = len(rec), len(rec) / max(rec.measured_fps() or rec.fps, 1e-6)
    path = rec.save(index, task, success=success)
    tag = "成功" if success else "保留(未标成功)"
    print(f"\n  存好 episode {index:04d} [{tag}]：{n} 帧 / {sec:.1f}s -> {path.name}")
    for t in notes:
        print(f"    ⚠ {t}")
    return index + 1


def _read_key(stdin):
    """读一个键。方向键是三字节转义序列，要一次读完，否则会被当成三个键。"""
    ch = stdin.read(1)
    if ch != "\x1b":
        return ch
    if select.select([stdin], [], [], 0.002)[0]:
        rest = stdin.read(1)
        if rest == "[" and select.select([stdin], [], [], 0.002)[0]:
            return {"D": "LEFT", "C": "RIGHT", "A": "UP", "B": "DOWN"}.get(stdin.read(1), "ESC")
    return "ESC"


def main():
    args = parse_args()
    T = args.period

    runlog = None
    if args.log and args.log != "none":
        path = (HERE / "logs" / f"servoj_{time.strftime('%Y%m%d_%H%M%S')}.log") if args.log == "auto" else Path(args.log)
        runlog = RunLog(path)
        runlog.line("ARGS", " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))
        print(f"运行日志 → {runlog.path}", flush=True)

    cfg = AR5Config(mock=False, real_hand_in_mock=False)
    cfg.use_hand = False
    cfg.arm.use_realtime = False          # 我们自己进 RT，不要驱动的 1kHz 线程
    cfg.arm.ip, cfg.arm.local_ip = args.robot_ip, args.local_ip
    if args.record:
        # 沿用 config 里的键名 cam_high（俯视）/ cam_front（侧视），命令行只改路径。
        # 不另起新名字：之前 297 条数据用的就是这两个名字，换名会让数据集对不上。
        if args.cam_top:
            cfg.cameras.sources["cam_high"] = args.cam_top
        if args.cam_front:
            cfg.cameras.sources["cam_front"] = args.cam_front
    else:
        cfg.cameras.sources = {}
    env = AR5Env(cfg)
    print("连接 AR5 ...", flush=True)
    env.connect()

    margin = np.array([args.ws_margin_mm, args.ws_margin_mm, args.ws_margin_z_mm]) / 1000.0
    ws_lo = np.asarray(cfg.arm.workspace_min, dtype=float) + margin
    ws_hi = np.asarray(cfg.arm.workspace_max, dtype=float) - margin
    j_lo = j_lo_pre = np.asarray(cfg.arm.joint_min, dtype=float) + cfg.arm.joint_limit_margin
    j_hi = j_hi_pre = np.asarray(cfg.arm.joint_max, dtype=float) - cfg.arm.joint_limit_margin

    if not args.no_home:
        print("  ⚠ 即将关节回位，3 秒内 Ctrl+C 可取消 ...", flush=True)
        time.sleep(3.0)
        print("  回位中 ...", end=" ", flush=True)
        print("完成" if env.home_joints() is not False else "未在超时内到位")

    if args.ik == "geo":
        from ar5_ik_select import AR5SelectiveIK
        ik = AR5SelectiveIK(joint_min=j_lo_pre, joint_max=j_hi_pre,
                            w_limit=args.w_limit, w_psi=args.w_psi,
                            psi_local=args.psi_span, max_jump=args.ik_max_jump,
                            max_step=args.max_step_rad)
        print(f"  逆解: 闭式解 + 选解层（跳变门 {np.rad2deg(args.ik_max_jump):.0f}° → "
              f"构型筛选 CFGX → 代价排序；照 robot_kindyn/ikgeo）"
              + ("" if ik.sew is not None else "  ⚠ 构型判别不可用"))
    elif args.ik == "poe":
        from ar5_poe_ik import AR5PoeIK
        ik = AR5PoeIK(max_step=args.max_step_rad)      # 自带 fk
        print("  逆解: 旋量法解析解 (AR5PoeIK，臂角跟着上一帧漂)")
    else:
        ik = env.arm.kinematics                        # 从控制器读 DH 构造
        print("  逆解: 驱动自带")
    fk = ik if hasattr(ik, "fk") else env.arm.kinematics
    # 自检：正解出来的 TCP 必须和控制器报的一致，否则映射基准就是错的
    q_chk = np.asarray(env.arm.get_joint_positions(), dtype=float)
    tcp_local = fk.fk(q_chk)[:3, 3]
    ec = {}
    cp = env.arm._robot.cartPosture(env.arm.sdk.CoordinateType.flangeInBase, ec)
    tcp_ctrl = np.asarray(cp.trans, dtype=float)
    err_mm = float(np.linalg.norm(tcp_local - tcp_ctrl)) * 1000
    print(f"  正解自检: 本地 {np.round(tcp_local*1000,1).tolist()} vs 控制器 "
          f"{np.round(tcp_ctrl*1000,1).tolist()} mm，差 {err_mm:.1f}mm")
    if err_mm > 5.0:
        print("  ✗ 正解和控制器对不上，映射基准会错，停止。检查 ar5_dh.json 或换 --ik driver")
        env.close()
        return 2

    streamer = connect_streamer(args)
    if streamer is None:
        env.close()
        return 1
    calib = Calibration.load(Path(args.calib_file)) if Path(args.calib_file).exists() else Calibration()
    hand = tracker = None
    try:
        hand = setup_hand(args, calib)
    except Exception as e:  # noqa: BLE001
        print("\n" + "!" * 62)
        print(f"!! 手没接上：{type(e).__name__}: {e}")
        print(f"!!   端口 {args.hand_port}")
        print("!!   查：ls -la /dev/serial/by-id/   fuser -v /dev/ttyUSB*")
        print("!! **本次只跑臂，手不会动**")
        print("!" * 62 + "\n")
        hand = None
    if hand is not None:
        retarget = Retargeter(calib, smooth=args.smooth, pinch_gate=args.pinch_gate)
        tracker = HandTracker(streamer, retarget, hand, args.hand_hz, args.freeze_thumb_rot, False)
        tracker.start()

    A = yaw_matrix(args.yaw) @ mirror_matrix(args.mirror)
    gate = JumpGate(args.hand_speed_max, T)
    kf = KalmanCV(args.kf_sigma_a, args.kf_sigma_m)
    lp_pos = OneEuro(args.lp_cutoff, args.lp_beta) if args.lp_cutoff > 0 else None
    lp_rot = OneEuro(args.lp_cutoff, args.lp_beta) if args.lp_cutoff > 0 else None
    tgt_filt_p = OneEuro(args.target_cutoff, args.target_beta) if args.target_cutoff > 0 else None
    tgt_filt_r = OneEuro(args.target_cutoff, args.target_beta) if args.target_cutoff > 0 else None

    print("\n" + "─" * 66)
    print(f"  臂 {args.robot_ip}  servoJ 覆盖式  T={T*1000:.0f}ms 前瞻={args.lookahead*1000:.0f}ms Kp={args.kp}")
    print(f"  预期滞后 ≈ {T*3*1000:.0f}ms（实测规律：滞后≈3×周期）")
    print(f"  滤波: 头显位姿 卡尔曼(σa={args.kf_sigma_a}) → One-Euro({args.lp_cutoff}Hz β={args.lp_beta})"
          + (f"  |  逆解前 目标位姿 One-Euro({args.target_cutoff}Hz β={args.target_beta})"
             if args.target_cutoff > 0 else "  |  逆解前 不再滤"))
    print(f"  缩放 x{args.scale}  yaw {args.yaw}°  姿态 {args.rot_mode}  单帧步长上限 {args.max_step_rad}rad")
    print(f"  工作空间 x[{ws_lo[0]:.2f},{ws_hi[0]:.2f}] y[{ws_lo[1]:.2f},{ws_hi[1]:.2f}] z[{ws_lo[2]:.2f},{ws_hi[2]:.2f}] m")
    print(f"  手: {'真手 ' + args.hand_port if hand is not None else '**无**（--no-hand 或连不上）'}")
    print("  ENTER 挂/摘离合   右手保持捏合驱动臂   h 停下回位   q 退出")
    print("─" * 66, flush=True)

    # ── 数采 ──
    rec = cams = None
    ep_index = n_saved = 0
    recording = False
    reset_until = 0.0
    if args.record:
        from episode_recorder import EpisodeRecorder, next_index
        cam_keys = list(cfg.cameras.sources.keys())
        if not cam_keys:
            print("  ✗ --record 但没有配相机，config.py 的 CameraConfig.sources 是空的")
            env.close()
            return 2
        cams = getattr(env, "cameras", None)
        if cams is None:
            print("  ✗ 相机没连上（AR5Env 没建 cameras）")
            env.close()
            return 2
        if not args.task:
            print("  ⚠ 没给 --task，数据里的任务描述是空的，训练时对不上语言条件")
        rec = EpisodeRecorder(args.out_dir, cam_keys, fps=args.record_hz, task=args.task)
        ep_index = next_index(args.out_dir)
        print(f"  数采: {rec.out_dir}  相机 {cam_keys}  {args.record_hz:.0f}Hz  "
              f"从 episode {ep_index:04d} 接着编"
              + (f"  目标 {args.num_episodes} 条" if args.num_episodes else "")
              + (f"  每条上限 {args.frame_cap} 帧" if args.frame_cap else ""))
        print("  键位: 空格=开始   s=存并标成功   ←=存但不标成功   退格=丢弃重录   q=退出"
              if not args.auto_record else "  接合即开录、摘离合即存盘")

    interactive = sys.stdin.isatty()
    old_term = termios.tcgetattr(sys.stdin) if interactive else None
    if interactive:
        tty.setcbreak(sys.stdin.fileno())

    armed = deadman = False
    ref = None
    prev_frame = None
    last_fresh = time.monotonic()
    last_print = last_snap = 0.0
    hand_v = np.zeros(3)
    filt_p = filt_R = None
    filt_t = None
    target_p = None
    n_frames = n_ik_fail = n_step_clip = n_ws_clip = n_rec_err = 0
    last_rec = 0.0
    rec_period = 1.0 / max(args.record_hz, 1.0)
    engaged_ok = False
    R_t = np.eye(3)
    ik_ms = []
    e2e_t, e2e_target, e2e_actual = [], [], []
    last_e2e = 0.0
    want_home = False

    try:
        with ServoJSession(env, T, args.lookahead, args.kp) as sj:
            q_cmd = sj.joints()
            q_goal = q_cmd.copy()
            last_goal_t = 0.0
            if args.planner == "none":
                trap = None                       # 第一版：没有规划器，只有步长限幅
                print("  规划器: **无**（第一版行为：逆解直接发 + 单帧步长限幅）")
            elif args.planner == "spline":
                trap = DelayedSpline(q_cmd, T, args.lag_ms / 1000.0, args.max_vel, args.max_acc)
                print(f"  规划器: 延迟缓冲 {args.lag_ms:.0f}ms + 穿点样条（内插，不外推）  "
                      f"发送周期 {T*1000:.0f}ms")
            elif args.planner == "cubic":
                trap = CubicInterp(q_cmd, T, args.max_vel, args.max_acc,
                                   feedforward=not args.no_ff, v_tau=args.v_tau)
                print(f"  规划器: 两点三次样条插值  插值周期 {T*1000:.0f}ms  "
                      f"v≤{args.max_vel} a≤{args.max_acc}  前馈{'关' if args.no_ff else '开'}"
                      f"  目标速度平滑 {args.v_tau*1000:.0f}ms")
            elif args.planner == "ruckig":
                try:
                    trap = RuckigTracker(q_cmd, T, args.max_vel, args.max_acc, args.max_jerk,
                                         feedforward=not args.no_ff)
                    print(f"  规划器: Ruckig S曲线  v≤{args.max_vel} a≤{args.max_acc} "
                          f"jerk≤{args.max_jerk} rad/s³  前馈{'关' if args.no_ff else '开'}")
                except Exception as e:  # noqa: BLE001
                    print(f"  规划器: ruckig 不可用（{e}），退回梯形")
                    trap = TrapTracker(q_cmd, T, args.max_vel, args.max_acc, args.brake)
            else:
                trap = TrapTracker(q_cmd, T, args.max_vel, args.max_acc, args.brake)
                print(f"  规划器: 梯形  v≤{args.max_vel} a≤{args.max_acc} 刹车系数{args.brake}")
            if runlog:
                runlog.line("INIT", f"q0={np.round(np.rad2deg(q_cmd),2).tolist()}deg")
            nxt = time.monotonic()
            while True:
                t0 = time.monotonic()

                if interactive and select.select([sys.stdin], [], [], 0)[0]:
                    ch = _read_key(sys.stdin)
                    if ch == "q":
                        break
                    elif rec is not None and ch in (" ", "s", "LEFT", "\x7f", "\b"):
                        if ch == " ":                       # 空格：开始录
                            if recording:
                                print("\n  已经在录了，按 s 存、← 保留、退格丢弃")
                            elif time.monotonic() < reset_until:
                                print(f"\n  复位中，还有 {reset_until - time.monotonic():.0f}s")
                            elif not engaged_ok:
                                print("\n  先挂离合并捏合右手，再按空格")
                            else:
                                rec.reset()
                                recording = True
                                cap = f"（上限 {args.frame_cap} 帧）" if args.frame_cap else ""
                                print(f"\n  ● 开始录 episode {ep_index:04d}{cap}"
                                      f"   s=成功  ←=保留  退格=丢弃")
                        elif not recording:
                            print("\n  还没开始录，按空格")
                        elif ch in ("s", "LEFT"):           # s=成功  ←=保留
                            recording = False
                            ep_index = _end_episode(rec, ep_index, args.task, success=(ch == "s"))
                            n_saved += 1
                            reset_until = time.monotonic() + args.reset_seconds
                            if args.reset_seconds > 0:
                                print(f"    复位 {args.reset_seconds:.0f}s，摆好现场再按空格")
                            if args.num_episodes and n_saved >= args.num_episodes:
                                print(f"\n  已采够 {n_saved} 条，收工")
                                break
                        else:                               # 退格：丢弃
                            recording = False
                            print(f"\n  ✗ 丢弃这条，编号仍是 {ep_index:04d}，摆好现场按空格重录")
                            rec.reset()
                    elif ch in ("\r", "\n"):
                        armed = not armed
                        ref = None
                        if runlog:
                            runlog.line("CLUTCH", "armed" if armed else "disarmed")
                        print(f"\n[离合] {'挂上' if armed else '摘下，臂保持当前位置'}")
                    elif ch == "h":
                        armed, ref, want_home = False, None, True
                        print("\n[回位] 退出 servoJ 后回位 ...")
                        break

                frame = streamer.latest
                fresh = frame is not None and frame is not prev_frame
                data = streamer.get_latest()
                if fresh:
                    prev_frame, last_fresh = frame, t0
                    n_frames += 1
                stale = (t0 - last_fresh) > args.max_stale
                if stale and armed:
                    print(f"\n[链路] 超过 {args.max_stale}s 没有新帧，摘离合")
                    if runlog:
                        runlog.line("ERR", "headset stale → disarm")
                    armed, ref = False, None

                if data is not None:
                    r_pinch = float(data["right_pinch_distance"])
                    deadman = r_pinch < (PINCH_OFF if deadman else PINCH_ON)
                    if fresh:
                        raw_p = wrist_xyz(data, "left")
                        raw_R = wrist_R(data, "left")
                        p_g, ok = gate(raw_p, t0)
                        p_k, hand_v = kf(p_g, t0)
                        if lp_pos is not None:
                            filt_p = lp_pos(p_k, t0)
                            filt_R = filter_rotation(lp_rot, raw_R, t0) if (ok or filt_R is None) else filt_R
                        else:
                            filt_p, filt_R = p_k, raw_R
                        filt_t = t0
                else:
                    r_pinch = 1.0

                engaged = armed and deadman and not stale and filt_p is not None
                if not engaged:
                    ref = None
                elif ref is None:
                    q_now = sj.joints()
                    T_now = fk.fk(q_now) if hasattr(fk, "fk") else None
                    if T_now is None:
                        engaged = False
                    else:
                        ref = (filt_p.copy(), filt_R.copy(), T_now[:3, 3].copy(), T_now[:3, :3].copy())
                        q_cmd = q_now.copy()
                        q_goal = q_now.copy()
                        if trap is not None:
                            trap.reset(q_now)
                        if hasattr(ik, "anchor"):       # 接合时把臂角锚在当前位形
                            psi0 = ik.anchor(q_now)
                            if runlog and psi0 is not None:
                                runlog.line("CLUTCH", f"臂角锚定 ψ={np.rad2deg(psi0):.1f}°")
                        print("\n[接合] 已锁定基准")
                        if runlog:
                            runlog.line("CLUTCH", f"engaged tcp={np.round(T_now[:3,3]*1000,1).tolist()}mm")

                # ── 目标位姿 → 逆解 → 限幅 → 发送 ──
                if engaged and ref is not None:
                    h0, R0, a0, Ra0 = ref
                    # 帧间预测：头显只有 ~60Hz，而控制周期是 8ms，中间两个周期若沿用
                    # 同一个陈旧目标，规划器的速度前馈就按错误的间隔算（实测差 3 倍）。
                    # 卡尔曼已经在估手速，用它把位置外推到当前时刻，目标就以控制周期连续变化。
                    hand_now = filt_p
                    if args.predict and filt_t is not None:
                        dtp = float(np.clip(t0 - filt_t, 0.0, args.predict_max))
                        hand_now = filt_p + hand_v * dtp
                    target_p = a0 + A @ (hand_now - h0) * args.scale
                    clipped = np.clip(target_p, ws_lo, ws_hi)
                    if not np.allclose(clipped, target_p):
                        n_ws_clip += 1
                    target_p = clipped
                    if args.rot_mode == "none":
                        R_t = Ra0
                    else:
                        rel = A @ (filt_R @ R0.T) @ A.T
                        rot = matrix_to_axis_angle(rel) * args.rot_scale
                        if args.rot_mode == "roll":
                            axis = Ra0[:, 2]
                            rot = axis * float(rot @ axis)
                        R_t = axis_angle_to_matrix(rot) @ Ra0
                    # ── 进逆解前的最后一级滤波 ──
                    # 前面的滤波都在头显原始位姿上，但到这里又经过了两步会重新引入
                    # 噪声的运算：帧间预测加的是卡尔曼估的手速×时间（手速本身有噪声），
                    # 以及坐标映射和 roll 模式的轴投影。这一级直接滤**送进逆解的目标**，
                    # 位置和姿态分开滤，姿态取前两列滤完再 Gram-Schmidt 重建
                    # （九个元素分别滤出来不是正交阵，送进逆解是个带缩放的伪旋转）。
                    if tgt_filt_p is not None:
                        target_p = tgt_filt_p(target_p, t0)
                        R_t = filter_rotation(tgt_filt_r, R_t, t0)
                    T_t = np.eye(4)
                    T_t[:3, :3], T_t[:3, 3] = R_t, target_p
                    t_ik = time.monotonic()
                    q_new, info = ik.ik(T_t, q_cmd)
                    ik_ms.append((time.monotonic() - t_ik) * 1000)
                    if q_new is None or not np.all(np.isfinite(q_new)):
                        n_ik_fail += 1
                    else:
                        q_goal = np.asarray(q_new, dtype=float)
                    # 规划/插值：cubic 只在拿到新逆解时重拟合，其余周期沿段推进；
                    # ruckig/trap 每周期都朝最新目标重规划
                    q_prev = q_cmd.copy()
                    if trap is None:                       # 第一版：直接取逆解结果
                        q_cmd = np.clip(q_goal, j_lo, j_hi)
                    elif isinstance(trap, DelayedSpline):
                        if fresh:
                            trap.push(q_goal, t0, j_lo, j_hi)
                        q_cmd = trap.step(t0, j_lo, j_hi).copy()
                    elif isinstance(trap, CubicInterp):
                        if fresh:
                            dt_goal = t0 - last_goal_t if last_goal_t else T
                            last_goal_t = t0
                            trap.set_goal(q_goal, float(np.clip(dt_goal, T, 0.1)), j_lo, j_hi)
                        q_cmd = trap.step(lo=j_lo, hi=j_hi).copy()
                    else:
                        q_cmd = trap.step(q_goal, j_lo, j_hi).copy()
                    m = float(np.max(np.abs(q_cmd - q_prev)))
                    if m > args.max_step_rad:                 # 硬上限兜底
                        q_cmd = q_prev + (q_cmd - q_prev) * (args.max_step_rad / m)
                        if trap is not None:
                            trap.q = q_cmd.copy()
                        n_step_clip += 1
                elif trap is not None:
                    trap.reset(q_cmd)                          # 不接合：速度归零，下次从静止起步

                try:
                    sj.send(q_cmd)            # 不接合时也持续发当前值，保持流不断
                except Exception as e:  # noqa: BLE001
                    print(f"\n[servoJ] 发送失败: {e}")
                    if runlog:
                        runlog.line("ERR", f"sendCommand: {e}")
                    break

                # ── 数采：按 record-hz 取一帧 ──
                engaged_ok = bool(engaged and ref is not None)
                if rec is not None:
                    if args.auto_record:
                        if engaged_ok and not recording:
                            rec.reset()
                            recording = True
                            print(f"\n  ● 开始录 episode {ep_index:04d}")
                        elif recording and not engaged_ok:
                            recording = False
                            ep_index = _end_episode(rec, ep_index, args.task)
                            n_saved += 1
                    # 帧数上限：按帧数不按墙上时间，回路周期抖一下不该改变一条的长短
                    if recording and args.frame_cap and len(rec) >= args.frame_cap:
                        recording = False
                        print(f"\n  到帧数上限 {args.frame_cap}，自动存盘")
                        ep_index = _end_episode(rec, ep_index, args.task, success=True)
                        n_saved += 1
                        reset_until = time.monotonic() + args.reset_seconds
                        if args.num_episodes and n_saved >= args.num_episodes:
                            print(f"\n  已采够 {n_saved} 条，收工")
                            break
                    if recording and t0 - last_rec >= rec_period:
                        last_rec = t0
                        try:
                            q_meas = sj.joints()
                            Tm = fk.fk(q_meas)
                            ee = np.concatenate([Tm[:3, 3], matrix_to_axis_angle(Tm[:3, :3])])
                            ee_t = (np.concatenate([target_p, matrix_to_axis_angle(R_t)])
                                    if target_p is not None else ee)
                            h_cmd = h_meas = None
                            if tracker is not None:
                                h_cmd, h_meas, _ = tracker.snapshot_full()
                            rec.add(q_meas, q_cmd, h_meas, h_cmd, ee, ee_t, cams.read())
                        except Exception as e:      # noqa: BLE001
                            n_rec_err += 1
                            if n_rec_err <= 3:
                                print(f"\n  [录制] 取帧失败: {type(e).__name__}: {e}")

                if engaged and target_p is not None and t0 - last_e2e >= 0.02:
                    last_e2e = t0
                    try:
                        Tn = fk.fk(sj.joints())
                        e2e_t.append(t0)
                        e2e_target.append(target_p.copy())
                        e2e_actual.append(Tn[:3, 3].copy())
                    except Exception:  # noqa: BLE001
                        pass

                if runlog and t0 - last_snap >= 1.0:
                    last_snap = t0
                    runlog.line("SNAP", f"engaged={int(bool(engaged and ref is not None))} "
                                f"手速={np.linalg.norm(hand_v)*1000:.0f}mm/s 发送={sj.sent} 迟到={sj.late} "
                                f"ik失败={n_ik_fail} 步长限幅={n_step_clip} 盒钳位={n_ws_clip} "
                                f"ik中位={np.median(ik_ms) if ik_ms else float('nan'):.1f}ms "
                                f"q={np.round(np.rad2deg(q_cmd),1).tolist()}")

                if args.print_hz > 0 and t0 - last_print >= 1.0 / args.print_hz:
                    last_print = t0
                    try:
                        Tn = fk.fk(sj.joints())
                        tcp = Tn[:3, 3]
                        lag = np.linalg.norm(target_p - tcp) * 1000 if target_p is not None else 0.0
                    except Exception:  # noqa: BLE001
                        tcp, lag = np.zeros(3), 0.0
                    tag = "接合" if engaged and ref is not None else ("等右手捏合" if armed else "未挂离合")
                    h_txt = ""
                    if tracker is not None:
                        h_txt = (f" | 手 写{tracker.ticks} 错{tracker.errors}"
                                 + (f" {tracker.last_error[:28]}" if tracker.errors else ""))
                    print(f"\r[{tag:^10}] R捏={r_pinch:.3f} 末端({tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f}) "
                          f"落后 {lag:5.1f}mm 手速 {np.linalg.norm(hand_v)*1000:4.0f}mm/s "
                          f"| 发{sj.sent} ik失败{n_ik_fail} 限幅{n_step_clip}{h_txt}   ",
                          end="", flush=True)

                nxt += T
                s = nxt - time.monotonic()
                if s > 0:
                    time.sleep(s)
                else:
                    sj.late += 1
                    nxt = time.monotonic()
    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        if old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        print("\n收尾 ...")
        if rec is not None:
            if recording and len(rec):
                # 中途退出：这条没人按 s，不敢当成功，但也不扔
                ep_index = _end_episode(rec, ep_index, args.task, success=False)
                n_saved += 1
            rec.flush()
            print(f"  数采: 这次存了 {n_saved} 条，目录里编到 {ep_index - 1:04d}，在 {rec.out_dir}"
                  + (f"  取帧失败 {n_rec_err} 次" if n_rec_err else ""))
        if tracker is not None:
            print(f"  手: 跟踪 {tracker.ticks} 次  出错 {tracker.errors} 次"
                  + (f"  最后一次: {tracker.last_error}" if tracker.errors else "")
                  + f"  实测 {getattr(tracker,'hz_est',0):.0f}Hz")
            if tracker.ticks == 0:
                print("     ⚠ 一次都没跟踪到 —— 头显没给手指数据，或 retarget 出错")
            tracker.stop()
            tracker.join(timeout=1.0)
        if hand is not None:
            try:
                hand.stop_writer()
                hand.close()
            except Exception:  # noqa: BLE001
                pass
        if want_home:
            try:
                print("  关节回位 ...", end=" ", flush=True)
                print("完成" if env.home_joints() is not False else "超时")
            except Exception as e:  # noqa: BLE001
                print(f"失败: {e}")
        print(f"  头显帧 {n_frames}  逆解 中位 {np.median(ik_ms) if ik_ms else float('nan'):.2f}ms "
              f"p95 {np.percentile(ik_ms,95) if ik_ms else float('nan'):.2f}ms  失败 {n_ik_fail}")
        print(f"  单帧步长限幅 {n_step_clip} 次  工作空间钳位 {n_ws_clip} 次")
        if hasattr(ik, "report"):
            print("  " + ik.report())
        try:
            if trap is not None:
                print(f"  规划器({args.planner}): 速度饱和 {trap.n_vel_sat} 周期（可提 --max-vel）  "
                      f"加速度饱和 {trap.n_acc_sat} 周期（可提 --max-acc）")
        except NameError:
            pass
        txt = e2e_report(e2e_t, e2e_target, e2e_actual)
        print(txt)
        if runlog:
            runlog.line("STAT", f"frames={n_frames} ik_fail={n_ik_fail} step_clip={n_step_clip} ws_clip={n_ws_clip}")
            runlog.line("STAT", txt.strip())
            runlog.close()
            print(f"  运行日志 {runlog.path}")
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
