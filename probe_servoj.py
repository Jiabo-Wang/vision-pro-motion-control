#!/usr/bin/env python3
"""量 AR5 的 servoJ：覆盖式关节目标流，控制器插补。独立脚本，不碰遥操代码。

servoJ 和 MoveL 的区别：MoveL 每个点是必须走完的任务，执行时间 ≈ 2√(L/a)+74ms，
遥操延迟随手速增长（实测 240~500ms）。servoJ 每 T 秒给一个**关节角目标**，控制器
按前瞻时间插补到 1ms，新目标直接盖掉旧目标，没有队列、没有任务开销。

接口（motioncontrolRT.PyRTmotioncontrol7）：
    setServoJoint(T, lookahead, Kp, ec)  → startMove(jointPosition)
    → 每 T 秒 sendCommand(JointPosition(q))  → 结束 stopServoJoint()

三个测试，默认全跑：
  1. sine  单关节正弦跟随：量指令→实测的相位滞后，这就是 servoJ 的延迟
  2. step  阶跃：目标突变，量到位时间和超调
  3. rate  发送周期扫描：T 取 8/16/33ms，看哪个最稳、丢包与否

**臂会动**。默认只动 j0（底座旋转），幅度 ±5°，从当前姿态出发。
运行前确认臂周围无人无物。

用法：
    python probe_servoj.py                       # 全部测试，j0 ±5°
    python probe_servoj.py --joint 3 --amp 3     # 换关节、改幅度
    python probe_servoj.py --test sine --kp 2    # 只跑正弦，调增益
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/crp-5070ti-01/yuhang_workspace/vel_ar5/openvla-energy")

from experiments.robot.ar5.ar5_env import AR5Env      # noqa: E402
from experiments.robot.ar5.config import AR5Config    # noqa: E402


def check(ec, what):
    if ec.get("ec", 0):
        raise RuntimeError(f"{what}: {ec}")


class ServoJ:
    """servoJ 会话。进入时切 RT 模式并开启 servoJ，退出时一定还原成 NRT。"""

    def __init__(self, env, period, lookahead, kp, log=print):
        self.env, self.log = env, log
        self.r, self.sdk = env.arm._robot, env.arm.sdk
        self.period, self.lookahead, self.kp = period, lookahead, kp
        self.rt = None
        self.streaming = False
        self.servo_on = False
        self.moving = False
        self.sent = self.late = 0

    def call(self, fn, *a):
        ec = {}
        out = fn(*a, ec)
        check(ec, getattr(fn, "__name__", "sdk"))
        return out

    def joints(self):
        return np.asarray(self.call(self.r.jointPos), dtype=float)

    def __enter__(self):
        s, r = self.sdk, self.r
        self.call(r.setMotionControlMode, s.MotionControlMode.NrtCommandMode)
        # 退出 RT 后电机是掉电的，而且控制器要缓一下才接受上电。
        # 实测第二次进 servoJ 会话时一次上电不成功，要等 idle 再重试。
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
                    break
                time.sleep(0.1)
            if str(self.call(r.powerState)).endswith(".on"):
                break
            self.log(f"[servoJ] 上电未成功（第 {attempt+1} 次），等 1s 重试 "
                     f"状态={self.call(r.powerState)} {self.call(r.operationState)}")
            time.sleep(1.0)
        else:
            raise RuntimeError(f"电机上不了电: {self.call(r.powerState)}")
        # 实时状态流：不订阅的话 jointPos 返回的是不刷新的快照（驱动里踩过）
        import datetime
        r.startReceiveRobotState(datetime.timedelta(milliseconds=1), ["q_m"])
        self.streaming = True
        self.call(r.setMotionControlMode, s.MotionControlMode.RtCommandMode)
        self.rt = r.getRtMotionController()
        ec = {}
        self.rt.setServoJoint(float(self.period), float(self.lookahead), float(self.kp), ec)
        check(ec, "setServoJoint")
        self.servo_on = True
        self.rt.startMove(s.RtControllerMode.jointPosition)
        self.moving = True
        self.log(f"[servoJ] 已开启 T={self.period*1000:.0f}ms 前瞻={self.lookahead*1000:.0f}ms Kp={self.kp}")
        return self

    def send(self, q):
        self.rt.sendCommand(self.sdk.JointPosition([float(v) for v in q]))
        self.sent += 1

    def __exit__(self, *exc):
        s, r = self.sdk, self.r
        try:
            if self.moving:
                try:                       # 最后一条带 finished，让控制器干净收尾
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
            time.sleep(0.5)          # 给控制器缓冲，下一个会话才能顺利上电
            self.log(f"[servoJ] 已关闭，发送 {self.sent} 条，迟到 {self.late} 条")
        return False


def lag_by_correlation(t, cmd, act, max_s=0.6):
    """把实测曲线往前平移，找和指令最贴合的偏移量 = 跟随滞后。"""
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], dt)
    c = np.interp(grid, t, cmd)
    a = np.interp(grid, t, act)
    best, best_err = 0.0, np.inf
    for k in range(int(max_s / dt)):
        e = float(np.mean(np.abs(a[k:] - c[:len(grid) - k])))
        if e < best_err:
            best_err, best = e, k * dt
    return best, best_err


def run_sine(sj, j, amp_rad, freq, seconds, log=print):
    q0 = sj.joints()
    t0 = time.monotonic()
    rec = []
    nxt = t0
    while True:
        now = time.monotonic()
        el = now - t0
        if el > seconds:
            break
        q = q0.copy()
        q[j] = q0[j] + amp_rad * math.sin(2 * math.pi * freq * el)
        sj.send(q)
        rec.append((el, q[j], sj.joints()[j]))
        nxt += sj.period
        s = nxt - time.monotonic()
        if s > 0:
            time.sleep(s)
        else:
            sj.late += 1
            nxt = time.monotonic()
    a = np.asarray(rec)
    lag, err = lag_by_correlation(a[:, 0], a[:, 1], a[:, 2])
    amp_cmd = (a[:, 1].max() - a[:, 1].min()) / 2
    amp_act = (a[:, 2].max() - a[:, 2].min()) / 2
    log(f"  正弦 {freq:.2f}Hz 幅值 {np.rad2deg(amp_rad):.1f}°: "
        f"滞后 {lag*1000:4.0f}ms  幅值保持 {amp_act/max(amp_cmd,1e-9)*100:3.0f}%  "
        f"残差 {np.rad2deg(err):.3f}°  样本 {len(a)}")
    return lag


def run_step(sj, j, amp_rad, seconds, log=print):
    q0 = sj.joints()
    t0 = time.monotonic()
    rec = []
    nxt = t0
    while True:
        now = time.monotonic()
        el = now - t0
        if el > seconds:
            break
        q = q0.copy()
        q[j] = q0[j] + (amp_rad if el > seconds * 0.3 else 0.0)
        sj.send(q)
        rec.append((el, q[j], sj.joints()[j]))
        nxt += sj.period
        s = nxt - time.monotonic()
        if s > 0:
            time.sleep(s)
        else:
            sj.late += 1
            nxt = time.monotonic()
    a = np.asarray(rec)
    i0 = int(np.argmax(np.abs(np.diff(a[:, 1])) > 1e-6)) + 1
    t_step = a[i0, 0]
    target = a[-1, 1]
    start = a[i0 - 1, 2]
    span = target - start
    reach = None
    for k in range(i0, len(a)):
        if abs(a[k, 2] - target) < 0.02 * abs(span):
            reach = a[k, 0] - t_step
            break
    over = (np.max((a[i0:, 2] - start) / span) - 1.0) * 100 if abs(span) > 1e-9 else 0
    log(f"  阶跃 {np.rad2deg(amp_rad):.1f}°: 到位(2%) {reach*1000 if reach else float('nan'):4.0f}ms  "
        f"超调 {over:+.1f}%  末值误差 {np.rad2deg(abs(a[-1,2]-target)):.3f}°")


def main():
    p = argparse.ArgumentParser(description="量 servoJ 的跟随滞后和稳定性")
    p.add_argument("--robot-ip", default="192.168.2.160")
    p.add_argument("--local-ip", default="192.168.2.222")
    p.add_argument("--joint", type=int, default=0, help="动哪个关节 0..6")
    p.add_argument("--amp", type=float, default=5.0, help="幅值 度")
    p.add_argument("--period", type=float, default=0.02, help="servoJ 发送周期 s")
    p.add_argument("--lookahead", type=float, default=0.02, help="前瞻时间 s，0=按实际到达间隔算速度")
    p.add_argument("--kp", type=float, default=1.0, help="位置反馈增益")
    p.add_argument("--freqs", default="0.2,0.5,1.0", help="正弦频率 Hz")
    p.add_argument("--seconds", type=float, default=8.0, help="每个正弦跑多久")
    p.add_argument("--test", default="all", choices=("all", "sine", "step", "rate"))
    a = p.parse_args()

    cfg = AR5Config(mock=False, real_hand_in_mock=False)
    cfg.use_hand = False
    cfg.arm.use_realtime = False            # 我们自己进 RT，不要驱动的 1kHz 线程
    cfg.arm.ip, cfg.arm.local_ip = a.robot_ip, a.local_ip
    cfg.cameras.sources = {}
    env = AR5Env(cfg)
    print("连接 AR5 ...", flush=True)
    env.connect()
    amp = math.radians(a.amp)
    lo = np.asarray(cfg.arm.joint_min) + cfg.arm.joint_limit_margin
    hi = np.asarray(cfg.arm.joint_max) - cfg.arm.joint_limit_margin
    q = np.asarray(env.arm.get_joint_positions(), dtype=float)
    if not (lo[a.joint] <= q[a.joint] - amp and q[a.joint] + amp <= hi[a.joint]):
        print(f"j{a.joint} 当前 {np.rad2deg(q[a.joint]):.1f}°，±{a.amp}° 会超软限位，换关节或减小幅值")
        env.close()
        return 1
    print(f"⚠ 臂会动 j{a.joint}，幅值 ±{a.amp}°。3 秒内 Ctrl+C 取消 ...", flush=True)
    time.sleep(3)

    try:
        if a.test in ("all", "sine", "step"):
            # 正弦和阶跃合用一个会话：每次进出 RT 都要重新上电，能省则省
            print(f"\n【跟随测试】T={a.period*1000:.0f}ms 前瞻={a.lookahead*1000:.0f}ms Kp={a.kp}")
            with ServoJ(env, a.period, a.lookahead, a.kp) as sj:
                if a.test in ("all", "sine"):
                    for f in [float(x) for x in a.freqs.split(",")]:
                        run_sine(sj, a.joint, amp, f, a.seconds)
                if a.test in ("all", "step"):
                    run_step(sj, a.joint, amp, 4.0)
        if a.test in ("all", "rate"):
            print(f"\n【发送周期扫描】同一个 0.5Hz 正弦，看哪个周期最稳")
            for T in (0.008, 0.016, 0.033):
                with ServoJ(env, T, T, a.kp, log=lambda *_: None) as sj:
                    print(f"  T={T*1000:.0f}ms:", end=" ")
                    run_sine(sj, a.joint, amp, 0.5, 6.0)
        print("\n读法：滞后就是 servoJ 的端到端延迟（不含头显和滤波）。"
              "\n      幅值保持接近 100% 说明没有被削顶；残差大说明增益或前瞻要调。"
              "\n      和 MoveL 比：MoveL 在同等运动下是 240~500ms 且随手速增长。")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
