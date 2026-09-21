#!/usr/bin/env python3
"""量 AR5 在 NRT MoveL 下的真实速度和加减速，不跑遥操、不连头显。

问的是一个问题：一条 L 毫米的 MoveL 指令，设定速度 v，臂实际用多久走完，
以及这个时间里有没有匀速段。遥操延迟的下限完全由它决定。

做法：从当前位姿出发，沿 +Y 和 -Y 交替走若干条直线段，每条段长和速度不同，
以 1kHz 读 TCP（startReceiveRobotState + cartPosture），算出：
    实际耗时、峰值速度、达到峰值速度的时间、匀速段占比
再扫 adjustAcceleration 的几档，看加速度设定到底有没有用。

**臂会移动**。运行前确认周围无人无物，段长默认 ≤80mm，起点用当前位置。

用法：
    python probe_movel_speed.py                  # 默认扫描
    python probe_movel_speed.py --lengths 10,20,40,80 --speeds 100,300,600
    python probe_movel_speed.py --accel-scan     # 只扫加速度档位（固定 40mm/300）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/crp-5070ti-01/yuhang_workspace/vel_ar5/openvla-energy")

from experiments.robot.ar5.ar5_env import AR5Env      # noqa: E402
from experiments.robot.ar5.config import AR5Config    # noqa: E402


def ec_check(ec, what):
    if ec.get("ec", 0):
        raise RuntimeError(f"{what}: {ec}")


class Probe:
    def __init__(self, robot, sdk):
        self.r, self.sdk = robot, sdk
        self.cid = 0

    def call(self, fn, *a):
        ec = {}
        out = fn(*a, ec)
        ec_check(ec, getattr(fn, "__name__", "sdk"))
        return out

    def pose(self):
        cp = self.call(self.r.cartPosture, self.sdk.CoordinateType.flangeInBase)
        return np.asarray(cp.trans, dtype=float), list(cp.rpy), cp

    def idle(self):
        return self.call(self.r.operationState) == self.sdk.OperationState.idle

    def wait_idle(self, timeout=20.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.idle():
                return True
            time.sleep(0.005)
        return False

    def setup(self, accel=None, jerk=2.0):
        s, r = self.sdk, self.r
        self.call(r.setMotionControlMode, s.MotionControlMode.NrtCommandMode)
        self.call(r.setOperateMode, s.OperateMode.automatic)
        self.call(r.setPowerState, True)
        t0 = time.monotonic()
        while not str(self.call(r.powerState)).endswith(".on"):
            if time.monotonic() - t0 > 5:
                raise RuntimeError("电机上不了电")
            time.sleep(0.1)
        self.call(r.setDefaultConfOpt, False)
        self.call(r.setMaxCacheSize, 50)
        a0, j0 = s.PyTypeDouble(), s.PyTypeDouble()
        ec = {}
        r.getAcceleration(a0, j0, ec)
        rd = lambda x: x.get() if hasattr(x, "get") else getattr(x, "content", x)  # noqa: E731
        cur = (rd(a0), rd(j0))
        if accel is not None:
            ec = {}
            r.adjustAcceleration(float(accel), float(jerk), ec)
            ec_check(ec, "adjustAcceleration")
        self.call(r.moveReset)
        return cur

    def run_chain(self, start_p, rpy, cp, step, n, speed, zone=100.0, hz=500.0, separate=False):
        """n 个同向段。separate=False 一次 append 整个列表；True 分 n 次背靠背 append。

        这两种发法的差别就是遥操和探测脚本的差别：如果转弯区只在同一次提交的列表内部
        生效，分次发的总耗时会明显更长，那遥操就必须成批发。
        """
        cmds = []
        for i in range(n):
            t = self.sdk.CartesianPosition()
            t.trans = list(np.asarray(start_p) + np.asarray(step) * (i + 1))
            t.rpy = list(rpy)
            t.elbow, t.hasElbow, t.confData = cp.elbow, True, list(cp.confData)
            c = self.sdk.MoveLCommand(t)
            c.speed, c.zone = float(speed), float(zone)
            cmds.append(c)
        if separate:
            for c in cmds:
                self.cid += 1
                self.call(self.r.moveAppend, [c], self.sdk.PyString(str(self.cid)))
        else:
            self.cid += 1
            self.call(self.r.moveAppend, cmds, self.sdk.PyString(str(self.cid)))
        t_send = time.monotonic()
        ec = {}
        self.r.moveStart(ec)
        ts, ps = [], []
        deadline = t_send + 25.0
        period, nxt, seen = 1.0 / hz, time.monotonic(), False
        while time.monotonic() < deadline:
            p, _, _ = self.pose()
            ts.append(time.monotonic() - t_send)
            ps.append(p)
            st = self.call(self.r.operationState)
            if st == self.sdk.OperationState.moving:
                seen = True
            elif seen:
                break
            nxt += period
            s = nxt - time.monotonic()
            time.sleep(s) if s > 0 else None
            if s <= 0:
                nxt = time.monotonic()
        return np.asarray(ts), np.asarray(ps)

    def run_segment(self, start_p, rpy, cp, delta, speed, zone=100.0, hz=500.0):
        """发一条 MoveL，1/hz 秒采样一次 TCP，返回 (t[], p[])。"""
        target = self.sdk.CartesianPosition()
        target.trans = list(np.asarray(start_p) + np.asarray(delta))
        target.rpy = list(rpy)
        target.elbow, target.hasElbow, target.confData = cp.elbow, True, list(cp.confData)
        cmd = self.sdk.MoveLCommand(target)
        cmd.speed, cmd.zone = float(speed), float(zone)
        self.cid += 1
        ts, ps = [], []
        self.call(self.r.moveAppend, [cmd], self.sdk.PyString(str(self.cid)))
        t_send = time.monotonic()
        ec = {}
        self.r.moveStart(ec)                  # -20/768 都是良性
        deadline = t_send + 15.0
        period = 1.0 / hz
        nxt = time.monotonic()
        moving_seen = False
        while time.monotonic() < deadline:
            p, _, _ = self.pose()
            ts.append(time.monotonic() - t_send)
            ps.append(p)
            st = self.call(self.r.operationState)
            if st == self.sdk.OperationState.moving:
                moving_seen = True
            elif moving_seen:
                break
            nxt += period
            s = nxt - time.monotonic()
            if s > 0:
                time.sleep(s)
            else:
                nxt = time.monotonic()
        return np.asarray(ts), np.asarray(ps)


def analyse(ts, ps, L_mm, v_cmd):
    if len(ts) < 5:
        return None
    d = np.linalg.norm(ps - ps[0], axis=1) * 1000          # mm
    moved = d > 0.3
    if not moved.any():
        return dict(t_total=np.nan, v_peak=0, frac_cruise=0, reached=0)
    i0 = int(np.argmax(moved))
    i1 = len(d) - 1
    while i1 > i0 and abs(d[i1] - d[i1 - 1]) < 0.02:
        i1 -= 1
    t = ts[i0:i1 + 1] - ts[i0]
    dd = d[i0:i1 + 1]
    if len(t) < 3:
        return dict(t_total=np.nan, v_peak=0, frac_cruise=0, reached=dd[-1])
    v = np.gradient(dd, t)                                   # mm/s
    v_peak = float(np.percentile(v, 95))
    cruise = float((v > 0.9 * v_peak).mean())
    # 起步到达 90% 峰值用时 → 有效加速度
    k = np.argmax(v > 0.9 * v_peak)
    t_acc = float(t[k]) if k > 0 else float("nan")
    a_eff = (v_peak / 1000) / t_acc if t_acc and t_acc > 1e-3 else float("nan")
    return dict(t_total=float(ts[i1] - ts[i0]), t_start=float(ts[i0]), v_peak=v_peak,
                frac_cruise=cruise, reached=float(dd[-1]), t_acc=t_acc, a_eff=a_eff)


def main():
    ap = argparse.ArgumentParser(description="量 MoveL 的真实速度/加减速")
    ap.add_argument("--robot-ip", default="192.168.2.160")
    ap.add_argument("--local-ip", default="192.168.2.222")
    ap.add_argument("--lengths", default="10,20,40,80", help="段长 mm")
    ap.add_argument("--speeds", default="100,300,600", help="设定速度 mm/s")
    ap.add_argument("--accel", type=float, default=1.5, help="adjustAcceleration 的 acc")
    ap.add_argument("--jerk", type=float, default=2.0)
    ap.add_argument("--accel-scan", action="store_true", help="改为扫 acc 档位，固定 40mm/300mm/s")
    ap.add_argument("--chain", action="store_true",
                   help="连续串测：一次 append N 个同向段，测转弯区衔接下的持续速度（遥操真正的工况）")
    ap.add_argument("--chain-n", type=int, default=6)
    ap.add_argument("--chain-seg", default="10,20,40", help="串测的单段长度 mm")
    ap.add_argument("--compare", action="store_true",
                   help="串测时对照两种发送方式：一次 append 整个列表 vs 分 N 次背靠背 append")
    ap.add_argument("--zone", type=float, default=100.0)
    ap.add_argument("--hz", type=float, default=500.0, help="TCP 采样率")
    a = ap.parse_args()

    cfg = AR5Config(mock=False, real_hand_in_mock=False)
    cfg.use_hand = False
    cfg.arm.use_realtime = False
    cfg.arm.ip, cfg.arm.local_ip = a.robot_ip, a.local_ip
    cfg.cameras.sources = {}
    env = AR5Env(cfg)
    print("连接 AR5（NRT）...", flush=True)
    env.connect()
    pr = Probe(env.arm._robot, env.arm.sdk)
    print("⚠ 臂会沿 ±Y 往返移动，3 秒内 Ctrl+C 取消 ...", flush=True)
    time.sleep(3)

    try:
        cur = pr.setup(a.accel if not a.accel_scan else None, a.jerk)
        print(f"控制器当前 加速度/加加速度 百分比 = {cur}")
        p0, rpy, cp = pr.pose()
        print(f"起点 TCP {np.round(p0*1000,1).tolist()} mm\n")

        if a.chain:
            modes = [("一次发整表", False), ("分次背靠背", True)] if a.compare else [("一次发整表", False)]
            print(f"连续串测：{a.chain_n} 个同向段，转弯区 {a.zone}"
                  + ("   ← 对照两种发送方式" if a.compare else ""))
            print(f"{'发送方式':>10}{'单段mm':>7}{'段数':>5}{'设定v':>7} | {'总耗时ms':>9}{'每段ms':>8}"
                  f"{'峰值v':>7}{'匀速占比':>9}{'持续速度':>9}")
            print("-" * 74)
            sign = 1.0
            for seg in [float(x) for x in a.chain_seg.split(",")]:
                for v in [float(x) for x in a.speeds.split(",")]:
                    for name, sep in modes:
                        pr.wait_idle()
                        p, rpy, cp = pr.pose()
                        step = np.array([0.0, sign * seg / 1000.0, 0.0])
                        lo = np.asarray(cfg.arm.workspace_min) + 0.03
                        hi = np.asarray(cfg.arm.workspace_max) - 0.03
                        if np.any(p + step * a.chain_n < lo) or np.any(p + step * a.chain_n > hi):
                            sign = -sign
                            step = -step
                        ts, ps = pr.run_chain(p, rpy, cp, step, a.chain_n, v, a.zone, a.hz, sep)
                        r = analyse(ts, ps, seg * a.chain_n, v)
                        sign = -sign
                        if r is None or r["t_total"] != r["t_total"]:
                            print(f"{name:>10}{seg:7.0f}{a.chain_n:5d}{v:7.0f} | 采样不足")
                            continue
                        sustained = r["reached"] / r["t_total"]
                        print(f"{name:>10}{seg:7.0f}{a.chain_n:5d}{v:7.0f} | {r['t_total']*1000:9.0f}"
                              f"{r['t_total']*1000/a.chain_n:8.0f}{r['v_peak']:7.0f}"
                              f"{r['frac_cruise']*100:8.0f}%{sustained:9.0f}")
            print("\n读法：两种发送方式如果每段耗时差很多，说明转弯区只在**同一次 moveAppend 的列表内部**"
                  "\n      生效，跨调用的段各自减速到零 —— 那遥操就必须成批发，不能一次发一个点。")
            return 0

        combos = []
        if a.accel_scan:
            for acc in (0.2, 0.5, 1.0, 1.5):
                combos.append((40.0, 300.0, acc))
        else:
            for L in [float(x) for x in a.lengths.split(",")]:
                for v in [float(x) for x in a.speeds.split(",")]:
                    combos.append((L, v, a.accel))

        print(f"{'段长mm':>7}{'设定v':>7}{'acc':>5} | {'实际ms':>8}{'峰值v':>8}{'匀速占比':>9}{'到位mm':>8}{'加速ms':>8}{'a_eff m/s²':>11}")
        print("-" * 82)
        sign = 1.0
        last_acc = None
        for L, v, acc in combos:
            if acc != last_acc and a.accel_scan:
                ec = {}
                pr.r.adjustAcceleration(float(acc), float(a.jerk), ec)
                ec_check(ec, "adjustAcceleration")
                last_acc = acc
            pr.wait_idle()
            p, rpy, cp = pr.pose()
            delta = np.array([0.0, sign * L / 1000.0, 0.0])
            # 出界就换方向
            lo = np.asarray(cfg.arm.workspace_min) + 0.03
            hi = np.asarray(cfg.arm.workspace_max) - 0.03
            if np.any(p + delta < lo) or np.any(p + delta > hi):
                sign = -sign
                delta = -delta
            ts, ps = pr.run_segment(p, rpy, cp, delta, v, a.zone, a.hz)
            r = analyse(ts, ps, L, v)
            sign = -sign
            if r is None:
                print(f"{L:7.0f}{v:7.0f}{acc:5.1f} | 采样不足")
                continue
            print(f"{L:7.0f}{v:7.0f}{acc:5.1f} | {r['t_total']*1000:8.0f}{r['v_peak']:8.0f}"
                  f"{r['frac_cruise']*100:8.0f}%{r['reached']:8.1f}{r['t_acc']*1000 if r['t_acc']==r['t_acc'] else float('nan'):8.0f}"
                  f"{r['a_eff'] if r['a_eff']==r['a_eff'] else float('nan'):11.2f}")
        print("\n读法：匀速占比接近 0 说明整段都在加减速，段长再短也没用；"
              "\n      a_eff 就是控制器实际给的加速度，adjustAcceleration 若无效则各档 a_eff 相同。")
    finally:
        try:
            ec = {}
            pr.r.moveReset(ec)
        except Exception:  # noqa: BLE001
            pass
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
