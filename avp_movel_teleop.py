#!/usr/bin/env python3
"""Vision Pro → AR5 遥操，MoveL 多路点版（控制器规划，本地只做滤波）。

链路（每一帧）：
    头显手腕位姿 → 跳变门 → 卡尔曼(匀速模型, 顺带估手速) → One-Euro 低通
      → 绝对目标位姿(基座系) → 工作空间钳位
      → [--sample-hz] 相对队尾过了死区才进**本地路径队列**
      → [--flush-hz] 整个本地队列以一个列表 moveAppend([MoveLCommand,...]) 交给控制器
        每点段速 = clamp(max(手速×增益, 段长/采样周期), vmin, vmax)
        转弯区 = min(zone, 0.45×段长)，相邻段在区里混合不减速

和 avp_arm_teleop.py 的区别：
  * 不开 RT 流、没有本地 1kHz 跟踪器、没有缰绳/重锚/打滑。插补、限速、平滑全在控制器。
  * 不积压：段速按「一段在一个采样周期内走完」定，点进队列的速度与走掉的速度对齐；
    控制器在飞点数仍超过 --inflight 时这批只留最后一点并按积压比例提速追赶
    （目标是绝对位姿，丢中间点不会少走终点）。
  * 手停了就不再发点（死区），臂到最后一个点后停住。

已知代价（MoveL 的本性，不是 bug）：
  * 每段是一个任务，段短时加速度受限提不起速；段间靠 zone 混合，队列一空必减速到零。
  * 延迟 ≈ 一段的执行时间 + 一个下发周期。

按 ENTER 挂/摘离合，右手保持捏合才驱动臂；q 退出；h 停下并关节回位；[ ] 调死区。
"""
from __future__ import annotations

import argparse
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
from hand_retarget import Calibration, Retargeter  # noqa: E402
from vel_kin import matrix_to_rpy  # noqa: E402
from pose_math import rpy_matrix, rotation_distance  # noqa: E402
from experiments.robot.ar5.ar5_env import AR5Env  # noqa: E402
from experiments.robot.ar5.config import AR5Config  # noqa: E402
import inspire_hand6 as ih  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# 运行日志（发给别人看的最小集合）
# ──────────────────────────────────────────────────────────────────────────────
class PointTrace:
    """每个采样点在五个环节上的时间戳，一行一个点，退出时导出 CSV。

    环节：frame 头显帧到达 → filt 滤波完成 → queue 进本地队列 → push 交给控制器
          → done 控制器报告走完。全部是同一个单调时钟。
    """

    COLS = ("pid", "t_frame", "t_filt", "t_queue", "t_push", "t_done",
            "cid", "batch_size", "x", "y", "z", "seg_mm", "speed", "hand_mm_s",
            "inflight_at_push", "actual_x", "actual_y", "actual_z", "fate")

    def __init__(self):
        self.rows = {}          # pid -> dict
        self.next_pid = 1

    def new(self, **kw):
        pid = self.next_pid
        self.next_pid += 1
        row = {c: "" for c in self.COLS}
        row["pid"] = pid
        row["fate"] = "queued"
        row.update(kw)
        self.rows[pid] = row
        return pid

    def set(self, pid, **kw):
        r = self.rows.get(pid)
        if r is not None:
            r.update(kw)

    def dump(self, path, t0):
        import csv
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.COLS)
            for pid in sorted(self.rows):
                r = self.rows[pid]
                out = []
                for c in self.COLS:
                    v = r[c]
                    if c.startswith("t_") and isinstance(v, float):
                        v = round(v - t0, 4)
                    elif isinstance(v, float):
                        v = round(v, 4)
                    out.append(v)
                w.writerow(out)
        return path


class RunLog:
    """一行一条，相对时间 + 标签 + 内容。tag 约定：
    ARGS 参数 | INIT 启动状态 | EVT 原始完成事件 | PUSH 交付明细 | DONE 完成与延迟 |
    SNAP 每秒快照 | CLUTCH 离合 | ERR 错误 | STAT 退出统计
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w", buffering=1)
        self.t0 = time.monotonic()
        self.f.write(f"# avp_movel_teleop run log  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def line(self, tag, msg):
        try:
            self.f.write(f"{time.monotonic()-self.t0:9.3f} [{tag:<6}] {msg}\n")
        except Exception:  # noqa: BLE001
            pass

    def close(self):
        try:
            self.f.close()
        except Exception:  # noqa: BLE001
            pass


# ──────────────────────────────────────────────────────────────────────────────
# 滤波
# ──────────────────────────────────────────────────────────────────────────────
class KalmanCV:
    """3 维位置的匀速模型卡尔曼滤波。状态 [p(3), v(3)]。

    作用有两个：一是在低通之前先把量测噪声按运动学模型压一遍；二是顺带给出
    **手速估计 v**，段速要用它来匹配。sigma_a 是过程噪声（人手加速度的量级），
    sigma_m 是头显位置量测噪声（m）。sigma_a 越小越平滑但越信模型、转向越钝。
    """

    def __init__(self, sigma_a: float = 3.0, sigma_m: float = 0.003):
        self.sigma_a, self.sigma_m = float(sigma_a), float(sigma_m)
        self.x = None
        self.P = None
        self.t = None

    def reset(self):
        self.x = self.P = self.t = None

    def __call__(self, z: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=float).reshape(3)
        if self.x is None:
            self.x = np.concatenate([z, np.zeros(3)])
            self.P = np.diag([self.sigma_m**2] * 3 + [1.0] * 3)
            self.t = t
            return self.x[:3].copy(), self.x[3:].copy()
        dt = float(np.clip(t - self.t, 1e-3, 0.5))
        self.t = t
        I3 = np.eye(3)
        F = np.block([[I3, dt * I3], [np.zeros((3, 3)), I3]])
        qa = self.sigma_a**2
        Q = qa * np.block([[dt**4 / 4 * I3, dt**3 / 2 * I3], [dt**3 / 2 * I3, dt**2 * I3]])
        x = F @ self.x
        P = F @ self.P @ F.T + Q
        H = np.hstack([I3, np.zeros((3, 3))])
        R = self.sigma_m**2 * I3
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        y = z - H @ x
        self.x = x + K @ y
        self.P = (np.eye(6) - K @ H) @ P
        return self.x[:3].copy(), self.x[3:].copy()


class JumpGate:
    """单帧位移超过人手物理可能的速度 → 整帧丢弃（追踪故障 / Wi-Fi 攒帧）。"""

    def __init__(self, vmax: float = 1.0, period: float = 0.02, max_run: int = 5):
        self.vmax, self.period, self.max_run = vmax, period, max_run
        self.prev = None
        self.prev_t = 0.0
        self.run = 0
        self.rejected = 0
        self.max_mm = 0.0

    def __call__(self, p: np.ndarray, t: float) -> tuple[np.ndarray, bool]:
        if self.vmax <= 0:
            return p, True
        if self.prev is None:
            self.prev, self.prev_t = p.copy(), t
            return p, True
        dt = min(max(t - self.prev_t, self.period), 3 * self.period)
        d = float(np.linalg.norm(p - self.prev))
        if d > self.vmax * dt and self.run < self.max_run:
            self.run += 1
            self.rejected += 1
            self.max_mm = max(self.max_mm, d * 1000)
            return self.prev.copy(), False
        self.run = 0
        self.prev, self.prev_t = p.copy(), t
        return p, True


# ──────────────────────────────────────────────────────────────────────────────
# MoveL 队列（只留最新目标）
# ──────────────────────────────────────────────────────────────────────────────
class MoveLQueue:
    """把「最新目标」以 MoveL 列表追加给控制器，并把在飞的点数压在 inflight 以内。

    在飞点数用 moveExecution 事件里的 cmdID 算：每次 append 的 cmdID 单调递增，
    控制器 FIFO 执行，所以最近一次 reachTarget/finish 的 cmdID 之前的全部完成。
    """

    def __init__(self, robot, sdk, *, zone_mm: float, vmin: float, vmax: float,
                 inflight: int, catchup: float = 0.5, consume_margin_mm: float = 2.0, log=print):
        self.robot, self.sdk, self.log = robot, sdk, log
        self.zone, self.vmin, self.vmax = float(zone_mm), float(vmin), float(vmax)
        self.inflight_max = max(1, int(inflight))
        self.catchup = float(catchup)
        self.catchup_base = 0            # 在飞点数超过它才算积压（水位模式设为低水位）
        self.consume_margin = float(consume_margin_mm) / 1000.0
        self.cmd_speed = 0.0             # >0 时所有段用这个固定设定速度
        self.send_elbow = True           # 是否给控制器指定肘角/构型
        self.max_seg = 0.0               # 单段最大长度 m，0=不限
        self.accel_scale = None          # adjustAcceleration 的 acc 百分比，None=不改
        self.jerk_scale = 1.0
        self.runlog = None               # RunLog，可选
        self.trace = None                # PointTrace，可选
        self.batch_pids = {}             # cmdID -> [pid...]
        self.next_id = 1          # 下一批 append 用的 cmdID
        self.finished_id = 0      # 已确认执行完的最大 cmdID
        self.batch_sizes = {}     # cmdID -> 该批点数，用来折算在飞点数
        self.batch_points = {}    # cmdID -> [trans...]，位置兜底判消费用
        self.partial = None       # (cmdID, 已完成点数)：当前正在执行的批走到哪了
        self._last_done_point = None   # 位置兜底：最近一个判定已走完的路点
        self._ev_queue = []
        self._ev_lock = __import__("threading").Lock()
        self.watcher_ok = False
        self.cid_offset = None         # 控制器 cmdID 编号 − 本地批号
        self._ev_last = None
        self._ev_debug = 0
        # 生产/消费计时：cmdID -> (发出时刻, 点数)；完成记录 [(完成时刻, 点数, 发出→完成 s)]
        self.sent_log = {}
        self.prod_times = []      # 每个发出的点一个时间戳
        self.cons_log = []        # (t_done, npts, latency)
        self.elbow = None
        self.conf = None
        self.last_sent = None     # (trans, rpy)
        self.last_sent_t = None   # 上一个已发点对应的手部时间戳
        self.stats = {"appended": 0, "batches": 0, "batch_max": 0, "thinned": 0, "held_full": 0,
                      "held_short": 0, "events": 0, "pose_marks": 0, "seg_clipped": 0,
                      "starts": 0, "already_moving": 0, "queue_empty": 0,
                      "exec_errors": 0, "speed_sum": 0.0,
                      "seg_len_sum": 0.0, "max_inflight_seen": 0}
        self.last_error = ""

    # ── SDK 小工具 ────────────────────────────────────────────────────────
    def _call(self, fn, *args):
        ec = {}
        out = fn(*args, ec)
        if ec.get("ec", 0):
            raise RuntimeError(f"{getattr(fn, '__name__', 'SDK')}: {ec}")
        return out

    def power_state(self) -> str:
        return str(self._call(self.robot.powerState))

    def is_idle(self) -> bool:
        return self._call(self.robot.operationState) == self.sdk.OperationState.idle

    def is_moving(self) -> bool:
        return self._call(self.robot.operationState) == self.sdk.OperationState.moving

    def pose(self):
        cp = self._call(self.robot.cartPosture, self.sdk.CoordinateType.flangeInBase)
        return np.asarray(cp.trans, dtype=float), list(cp.rpy), cp

    # ── 配置 ──────────────────────────────────────────────────────────────
    def configure(self, cache: int = 50):
        sdk, r = self.sdk, self.robot
        self._call(r.setMotionControlMode, sdk.MotionControlMode.NrtCommandMode)
        self._call(r.setOperateMode, sdk.OperateMode.automatic)
        # 2026-09-20 教训：上电必须紧跟在切模式之后；中间插 toolset/set* 时
        # setPowerState 会变成空操作（41ms 返回 0 但电机不上）。
        self._call(r.setPowerState, True)
        deadline = time.monotonic() + 5.0
        while not self.power_state().endswith(".on"):
            if time.monotonic() > deadline:
                raise RuntimeError(f"电机上不了电: {self.power_state()}")
            time.sleep(0.1)
        self._call(r.setDefaultConfOpt, False)       # 逆解取离当前最近的解
        self._call(r.setMaxCacheSize, int(cache))
        # 加/减速度与加加速度：系统预设的百分比，acc ∈ [0.2,1.5]，jerk ∈ [0.1,2]。
        # 2026-09-21 实测 15mm 段按 200mm/s 理论 75ms、实际 286ms，短段全程在加减速；
        # 这是 MoveL 上唯一能缩短「每点时间」的旋钮。
        try:
            a0, j0 = sdk.PyTypeDouble(), sdk.PyTypeDouble()
            ec = {}
            r.getAcceleration(a0, j0, ec)
            rd = lambda x: x.get() if hasattr(x, "get") else getattr(x, "content", x)  # noqa: E731
            before = f"{rd(a0)}/{rd(j0)}" if not ec.get("ec") else f"读失败{ec}"
        except Exception as exc:  # noqa: BLE001
            before = f"读失败 {exc}"
        if self.accel_scale is not None:
            ec = {}
            r.adjustAcceleration(float(self.accel_scale), float(self.jerk_scale), ec)
            msg = f"加速度/加加速度 预设百分比 {before} → 设为 {self.accel_scale}/{self.jerk_scale} (ec={ec.get('ec',0)})"
        else:
            msg = f"加速度/加加速度 预设百分比 {before}（未改）"
        self.log(f"[MoveL] {msg}")
        if self.runlog:
            self.runlog.line("INIT", msg)
        self._call(r.moveReset)
        # 完成通知只走回调。2026-09-21 实测 queryEventInfo 轮询永远返回同一条
        # {cmdID:'', reachTarget:False}，一场 51 个点解出 0 条完成 → 在飞数卡死。
        # 回调在 SDK 线程执行，只把 dict 塞进队列，主线程再处理。
        self._ev_queue = []
        self._ev_lock = __import__("threading").Lock()
        try:
            ec = {}
            r.setEventWatcher(sdk.Event.moveExecution, self._on_event, ec)
            self.watcher_ok = not ec.get("ec", 0)
            if not self.watcher_ok:
                self.log(f"[MoveL] setEventWatcher 失败 {ec}，只能靠位置兜底判完成")
        except Exception as exc:  # noqa: BLE001
            self.watcher_ok = False
            self.log(f"[MoveL] setEventWatcher 异常 {exc}，只能靠位置兜底判完成")
        p, rpy, cp = self.pose()
        self.elbow, self.conf = cp.elbow, list(cp.confData)
        self.last_sent = (p.copy(), list(rpy))
        self._last_done_point = p.copy()
        self.log(f"[MoveL] 电机 on，TCP {np.round(p*1000,1).tolist()} mm，队列上限 {self.inflight_max} 点")

    # ── 事件 / 在飞点数 ────────────────────────────────────────────────────
    @staticmethod
    def _as_int(v):
        """取字符串末尾的整数。控制器回报的 cmdID 是自己的编号 'l#12'，不是我们传的串。"""
        try:
            import re
            m = re.search(r"(\d+)\s*$", str(v))
            return int(m.group(1)) if m else None
        except Exception:  # noqa: BLE001
            return None

    def _map_cid(self, num):
        """控制器编号 → 我们的批号。两者都按 moveAppend 调用顺序 +1，只差一个固定偏移，
        用第一条事件对齐：它对应的必然是当时最老的未完成批。"""
        if num is None:
            return None
        if self.cid_offset is None:
            oldest = min((c for c in self.batch_sizes if c > self.finished_id), default=None)
            if oldest is None:
                return None
            self.cid_offset = num - oldest
            if self.runlog:
                self.runlog.line("EVT", f"cmdID 偏移对齐: 控制器 {num} ↔ 本地批 {oldest} (offset {self.cid_offset})")
        cid = num - self.cid_offset
        return cid if (cid in self.batch_sizes or cid <= self.finished_id) else None

    @staticmethod
    def _as_bool(v):
        return str(v).strip().lower() in ("true", "1", "yes")

    def _on_event(self, ev):
        """SDK 线程回调：只入队，不做任何 SDK 调用或打印。"""
        try:
            with self._ev_lock:
                self._ev_queue.append((time.monotonic(), dict(ev) if isinstance(ev, dict) else ev))
                if len(self._ev_queue) > 500:
                    del self._ev_queue[:-500]
        except Exception:  # noqa: BLE001
            pass

    def poll(self):
        """处理回调收到的 moveExecution 事件；按批/按点推进消费计数；执行错误返回错误码。

        控制器对每个路点发一条 (cmdID, wayPointIndex, reachTarget) 通知（SDK 日志里的
        `Notified l#N [k] finish:true`），只通过 setEventWatcher 回调送达。
        """
        with self._ev_lock:
            batch, self._ev_queue = self._ev_queue, []
        code = 0
        for t_ev, ev in batch:
            c = self._handle_event(ev, t_ev)
            code = c or code
        return code

    def _handle_event(self, ev, t_ev):
        try:
            if isinstance(ev, dict):
                get = ev.get
            else:                                   # 不是 dict 就按属性取
                get = lambda k, d=None: getattr(ev, k, d)  # noqa: E731
            K = getattr(getattr(self.sdk, "EventInfoKey", None), "MoveExecution", None)
            k_id = getattr(K, "ID", "cmdID")
            k_reach = getattr(K, "ReachTarget", "reachTarget")
            k_wp = getattr(K, "WaypointIndex", "wayPointIndex")
            k_err = getattr(K, "Error", "error")
            if self._ev_debug < 20:
                self._ev_debug += 1
                if self._ev_debug <= 3:
                    self.log(f"\n[事件#{self._ev_debug}] {ev!r}"[:300])
                if self.runlog:
                    self.runlog.line("EVT", repr(ev)[:400])
            err = get(k_err, 0)
            code = err.get("ec", 0) if isinstance(err, dict) else (
                int(err) if isinstance(err, (int, float)) else (self._as_int(getattr(err, "value", 0)) or 0))
            if code:
                self.stats["exec_errors"] += 1
                self.last_error = f"{ev}"
                return code
            cid = self._map_cid(self._as_int(get(k_id, None)))
            wp = self._as_int(get(k_wp, None))
            done = self._as_bool(get(k_reach, False))
            if cid is not None and done:
                key = (cid, wp)
                if key != self._ev_last:
                    self._ev_last = key
                    self.stats["events"] += 1
                    size = self.batch_sizes.get(cid)
                    if size is None or wp is None or wp >= size - 1:
                        self._mark_finished(cid, now=t_ev)   # 该批最后一点（或没有点索引）
                    else:
                        self._mark_finished(cid - 1, now=t_ev)   # 之前的批一定完成了
                        self.partial = (cid, wp + 1)         # 本批已完成 wp+1 个点
        except Exception as exc:  # noqa: BLE001
            if self._ev_debug < 6:
                self._ev_debug += 1
                self.log(f"\n[事件解析失败] {type(exc).__name__}: {exc}")
        return 0

    def _progress_by_pose(self):
        """兜底：按 TCP 沿已发折线的投影判哪些路点已经走过。

        转弯区大时臂抄近路，不会经过路点附近，所以不能用「离路点多近」；改用
        「TCP 是否已越过路点 k 处、垂直于下一段方向的平面」（越过 = 沿下一段的投影 > 0），
        最后一个点没有下一段，才退回用距离（转弯区×1.2 + 余量）判。
        事件回调正常时这里基本不起作用，只是保险。
        """
        n_open = sum(1 for c in self.batch_sizes if c > self.finished_id)
        if n_open == 0:
            return
        p, _, _ = self.pose()
        # 把未完成的路点按顺序摊平：[(cid, k, wp, zone), ...]
        flat = []
        for cid in sorted(c for c in self.batch_points if c > self.finished_id):
            for k, (wp, zone_m) in enumerate(self.batch_points[cid]):
                flat.append((cid, k, wp, zone_m))
        if not flat:
            return
        # 跳过本批已确认完成的前几个点
        start = 0
        if self.partial is not None and self.partial[0] == flat[0][0]:
            start = min(self.partial[1], len(flat))
        prev = self._last_done_point
        passed = -1
        for i in range(start, len(flat)):
            cid, k, wp, zone_m = flat[i]
            if i + 1 < len(flat):
                nxt = flat[i + 1][2]
                seg_v = nxt - wp
                seg_l = float(np.linalg.norm(seg_v))
                # 越过路点 k：TCP 相对 wp 在下一段方向上的投影 > 0，且离折线不远（防误判）
                ahead = seg_l > 1e-6 and float((p - wp) @ seg_v) / seg_l > 0.0
                near = np.linalg.norm(p - wp) < max(seg_l, 0.015 + self.consume_margin) + 0.02
                ok = ahead and near
            else:
                # zone 现在是百分比档位(100)，不能当几何半径用；终点按 15mm+余量 判到
                ok = np.linalg.norm(p - wp) < 0.015 + self.consume_margin
            if ok:
                passed = i
                prev = wp
            else:
                break
        if passed < 0:
            return
        cid, k, wp, _ = flat[passed]
        self._last_done_point = prev
        size = self.batch_sizes.get(cid, k + 1)
        if k >= size - 1:
            self._mark_finished(cid)
            self.stats["pose_marks"] += 1
        else:
            self._mark_finished(cid - 1)
            self.partial = (cid, k + 1)
            self.stats["pose_marks"] += 1

    def _mark_finished(self, upto: int, now: float | None = None):
        """cmdID ≤ upto 的批全部完成（FIFO）。记消费时刻和发出→完成延迟。"""
        if upto <= self.finished_id:
            return
        now = time.monotonic() if now is None else now
        for cid in sorted(c for c in self.sent_log if self.finished_id < c <= upto):
            t_send, npts = self.sent_log.pop(cid)
            self.cons_log.append((now, npts, now - t_send))
            if self.trace is not None:
                try:
                    ap, _, _ = self.pose()
                except Exception:  # noqa: BLE001
                    ap = (None, None, None)
                for pid in self.batch_pids.pop(cid, []):
                    self.trace.set(pid, t_done=now, fate="done",
                                   actual_x=float(ap[0]), actual_y=float(ap[1]), actual_z=float(ap[2]))
            if self.runlog:
                self.runlog.line("DONE", f"cid={cid} pts={npts} 发出→完成 {1000*(now-t_send):.0f}ms")
        self.finished_id = upto

    def rates(self, window: float = 2.0):
        """最近 window 秒：生产点/s、消费点/s、平均等待 ms、每点执行 ms。"""
        now = time.monotonic()
        prod = sum(1 for t in self.prod_times if now - t <= window)
        recent = [(t, n, lat) for t, n, lat in self.cons_log if now - t <= window]
        cons = sum(n for _, n, _ in recent)
        wait_ms = 1000 * np.mean([lat for _, _, lat in recent]) if recent else float("nan")
        per_pt = float("nan")
        if len(recent) >= 2:
            span = recent[-1][0] - recent[0][0]
            pts = sum(n for _, n, _ in recent[1:])
            per_pt = 1000 * span / pts if pts else float("nan")
        return prod / window, cons / window, wait_ms, per_pt

    def summary(self):
        if not self.prod_times:
            return "  生产/消费: 没有发出任何点"
        t0, t1 = self.prod_times[0], self.prod_times[-1]
        span = max(t1 - t0, 1e-6)
        lines = [f"  生产: {len(self.prod_times)} 点 / {span:.1f}s = {len(self.prod_times)/span:.1f} 点/s"]
        if self.cons_log:
            cons_pts = sum(n for _, n, _ in self.cons_log)
            c0, c1 = self.cons_log[0][0], self.cons_log[-1][0]
            cspan = max(c1 - c0, 1e-6)
            lats = np.array([lat for _, _, lat in self.cons_log]) * 1000
            gaps = []
            for (ta, _, _), (tb, nb, _) in zip(self.cons_log, self.cons_log[1:]):
                if nb > 0:
                    gaps.append((tb - ta) / nb * 1000)
            gaps = np.array(gaps) if gaps else np.array([np.nan])
            lines.append(f"  消费: {cons_pts} 点 / {cspan:.1f}s = {cons_pts/cspan:.1f} 点/s"
                         f"   每点执行 中位 {np.nanmedian(gaps):.0f}ms  p95 {np.nanpercentile(gaps,95):.0f}ms")
            lines.append(f"  发出→完成 等待: 中位 {np.median(lats):.0f}ms  p95 {np.percentile(lats,95):.0f}ms"
                         f"  最大 {lats.max():.0f}ms")
        else:
            lines.append("  消费: 没收到任何 moveExecution 完成事件（cmdID/reachTarget 字段可能不匹配）")
        return "\n".join(lines)

    def inflight(self) -> int:
        """控制器里还没执行完的**点数**（按批的 cmdID 与每批点数折算，减去本批已完成的点）。"""
        self._progress_by_pose()
        n = sum(sz for cid, sz in self.batch_sizes.items() if cid > self.finished_id)
        if self.partial is not None and self.partial[0] > self.finished_id:
            n -= min(self.partial[1], self.batch_sizes.get(self.partial[0], 0))
        # 控制器 idle 说明队列一定空了，事件漏掉也不至于把在飞数卡死
        if n > 0 and self.is_idle():
            self._mark_finished(self.next_id - 1)
            n = 0
        for cid in [c for c in self.batch_sizes if c <= self.finished_id]:
            del self.batch_sizes[cid]
            self.batch_points.pop(cid, None)
        if self.partial is not None and self.partial[0] <= self.finished_id:
            self.partial = None
        self.stats["max_inflight_seen"] = max(self.stats["max_inflight_seen"], n)
        return n

    # ── 下发 ──────────────────────────────────────────────────────────────
    def _cmd(self, trans, rpy, speed):
        t = self.sdk.CartesianPosition()
        t.trans, t.rpy = [float(v) for v in trans], [float(v) for v in rpy]
        if self.conf is not None and self.send_elbow:
            t.elbow, t.hasElbow, t.confData = self.elbow, True, list(self.conf)
        c = self.sdk.MoveLCommand(t)
        c.speed, c.zone = float(speed), self.zone
        return c

    def push_batch(self, points, sample_period: float, gain: float, zone_frac: float = 0.45) -> str:
        """把本地路径队列里的点以**一个列表** moveAppend 交给控制器。

        points: [(trans, rpy, hand_speed_mm_s), ...]，按时间顺序。
        每个点单独给速度：v = clamp(max(手速×gain, 段长/采样周期), vmin, vmax)，
        保证一段大约在一个采样周期内走完 —— 点进来的速度和走掉的速度对齐，队列就不涨。
        转弯区取段长的 zone_frac，相邻段在区里混合、不减速。
        控制器里在飞的点已经太多时（落后了），只保留这批的最后一个点并把速度按积压比例
        提高，追上来；目标是绝对位姿，丢中间点不会少走终点。
        """
        if not points:
            return "空"
        n = self.inflight()
        if n >= self.inflight_max:
            # 硬顶：一个点也不追加。2026-09-20 实测只压批不限次时在飞冲到 151 点、
            # 积压 12 秒。调用方保留最新点等下一次。
            self.stats["held_full"] += 1
            return f"满{n}"
        if n + len(points) > self.inflight_max:
            self.stats["thinned"] += len(points) - 1
            points = points[-1:]
        # 追赶：控制器里已有 n 个点没走完，这批的速度按积压比例提高。
        # 2026-09-20 实测 0.25 太温和：臂只比手快一点，积压 4 点 ≈ 1.2s 延迟消不掉。
        # 高低水位模式下缓冲区里的点是有意维持的，不算积压；只对超过低水位的部分追赶。
        catchup = 1.0 + max(0, n - self.catchup_base) * self.catchup
        # 肘角必须跟着臂走。AR5 是 7 轴冗余臂，elbow 是那个多余自由度；
        # 2026-09-21 之前只在 configure 时取一次，TCP 移开 200mm 后还要满足启动时的肘角，
        # 臂只能扭着走 —— 实跑每段 250ms，而探测脚本（每次用新鲜位姿）只要 65~82ms。
        if self.send_elbow:
            try:
                _, _, cp_now = self.pose()
                self.elbow, self.conf = cp_now.elbow, list(cp_now.confData)
            except Exception:  # noqa: BLE001
                pass
        cmds, prev, t_prev = [], self.last_sent[0], self.last_sent_t
        total_len = 0.0
        zones_m = []
        segs_mm, speeds = [], []
        for trans, rpy, hv, t_pt, _pid in points:
            trans = np.asarray(trans, dtype=float)
            seg = float(np.linalg.norm(trans - prev))                      # m
            # 单段限长。2026-09-21 实测执行时间随段长陡增：10mm 段 91ms、50mm+ 段 545ms。
            # 队列满时大量采样点被丢，手继续动，下次能发时目标已经很远 → 一发就是长段
            # → 执行更久 → 丢更多 → 段更长，自我恶化。截断后剩下的路下一段接着走。
            if 0 < self.max_seg < seg:
                trans = prev + (trans - prev) * (self.max_seg / seg)
                seg = self.max_seg
                self.stats["seg_clipped"] += 1
            # 臂走这一段的时间 = 手产生这一段用的时间。段长 / 相邻两点真实时间差。
            # （之前按采样周期 20ms 算，10mm 段要 500mm/s，臂 30ms 冲完再干等下一批
            #  → 一段一段动。）手速估计只作 dt 异常时的兜底。
            dt = t_pt - t_prev if t_prev is not None else 0.0
            v_time = seg / dt * 1000.0 if dt > 5e-3 else hv
            # 取两者的较大值。只用 seg/dt 会被死区坑：被死区挡掉的帧不进队列，
            # 手慢挪时一个 10mm 段对应 0.5s 的手部时间，算出 24mm/s 被夹到下限，
            # 臂以 40mm/s 爬（2026-09-21 实测 60 点里 15 点撞下限、段速中位仅 73）。
            # hv 是卡尔曼每帧更新的即时手速（已乘缩放），不受死区影响。
            if self.cmd_speed > 0:
                # 固定给高速度。2026-09-21 实测(probe_movel_speed.py)：设定速度不只是上限，
                # 它还决定关节速度百分比和加速度档位（robot.h: <100→10%, 100~200→30%,
                # 200~500→50%, 500~800→80%）。同样 10mm 段，设 100 要 240ms、设 600 只要 162ms；
                # 80mm 段 906ms vs 364ms。按手速给段速 = 手慢时掉进最低档、臂爬着走。
                # 段长本身会限制臂能跑多快，所以高设定值不会让它冲过头。
                speed = float(self.cmd_speed)
            else:
                speed = float(np.clip(max(v_time, hv) * gain * catchup, self.vmin, self.vmax))
            # zone 不是几何半径而是分档百分比（robot.h setDefaultZone 注释）：
            #   <1 → 0%(fine 停车)  1~20 → 10%  20~60 → 30%  >60 → 100%
            # 2026-09-21 之前按 0.45×段长给（9mm 落在 10% 档），每个点几乎都停车再起步，
            # 这就是量到的「每点 110ms 固定开销」。要衔接就给 >60，与段长无关。
            zone = float(self.zone)
            c = self._cmd(trans, rpy, speed)
            c.zone = zone
            cmds.append(c)
            zones_m.append(zone / 1000.0)
            segs_mm.append(seg * 1000)
            speeds.append(speed)
            self.stats["speed_sum"] += speed
            self.stats["seg_len_sum"] += seg * 1000
            total_len += seg * 1000
            prev, t_prev = trans, t_pt
        cid = self.next_id
        self._call(self.robot.moveAppend, cmds, self.sdk.PyString(str(cid)))
        self.batch_sizes[cid] = len(cmds)
        self.batch_points[cid] = [(np.asarray(tr, dtype=float).copy(), z)
                                  for (tr, _, _, _, _), z in zip(points, zones_m)]
        t_send = time.monotonic()
        self.sent_log[cid] = (t_send, len(cmds))
        self.prod_times.extend([t_send] * len(cmds))
        if self.trace is not None:
            self.batch_pids[cid] = [p[4] for p in points]
            for (tr, _, hv, _, pid), sg, sp in zip(points, segs_mm, speeds):
                self.trace.set(pid, t_push=t_send, cid=cid, batch_size=len(cmds),
                               seg_mm=sg, speed=sp, hand_mm_s=hv,
                               inflight_at_push=n, fate="pushed")
        self.next_id += 1                                # 一批一个 cmdID（列表在控制器里是一条路径）
        self.stats["appended"] += len(cmds)
        self.stats["batches"] += 1
        self.stats["batch_max"] = max(self.stats["batch_max"], len(cmds))
        _, r, _, t_last, _ = points[-1]
        self.last_sent = (prev.copy(), list(r))       # prev = 截断后实际发出的终点
        self.last_sent_t = t_last
        status = f"批{len(cmds)} {total_len:.0f}mm v̄={self.stats['speed_sum']/max(1,self.stats['appended']):.0f} 飞{n+1}"
        started = ""
        if not self.is_moving():
            started = self._start()
            status += "+" + started
        if self.runlog:
            self.runlog.line("PUSH", f"cid={cid} pts={len(cmds)} 折线={total_len:.0f}mm "
                             f"v={[round(c.speed) for c in cmds]} zone={[round(c.zone,1) for c in cmds]} "
                             f"在飞(前)={n} 追赶x{catchup:.2f} {'start:'+started if started else '已在动'}")
        return status

    def _start(self) -> str:
        ec = {}
        self.robot.moveStart(ec)
        code = ec.get("ec", 0)
        if code == 0:
            self.stats["starts"] += 1
            return "start"
        if code == -20:
            self.stats["already_moving"] += 1
            return "已在动"
        if code == 768:
            self.stats["queue_empty"] += 1
            return "队列空"
        raise RuntimeError(f"moveStart: {ec}")

    def stop(self, wait_s: float = 3.0):
        """等控制器 idle 后清队列。运动中 moveReset 会被拒 -20，所以先等。"""
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and not self.is_idle():
            time.sleep(0.02)
        ec = {}
        self.robot.moveReset(ec)
        self._mark_finished(self.next_id - 1)
        self.batch_sizes.clear()
        self.batch_points.clear()
        self.partial = None
        p, rpy, _ = self.pose()
        self._last_done_point = p.copy()
        self.last_sent = (p.copy(), list(rpy))
        self.last_sent_t = None
        return ec.get("ec", 0)


# ──────────────────────────────────────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Vision Pro → AR5 遥操（MoveL 多路点，控制器规划）",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("ip", help="Vision Pro 的 IP")
    p.add_argument("--hz", type=float, default=100, help="读头显/滤波的循环频率")
    p.add_argument("--sample-hz", type=float, default=20,
                   help="从滤波后的目标里取路径点进本地队列的频率。段长≈手速×这个周期")
    p.add_argument("--flush-hz", type=float, default=10,
                   help="本地队列整批(列表 MoveL)交给控制器的频率。20Hz采样/10Hz交付 → 每批约2点")
    p.add_argument("--direct", action="store_true",
                   help="直通模式：不要本地队列。滤波+死区之后的点立刻单条 moveAppend 发出，"
                        "控制器在飞数达到 --queue-high 就丢掉这个点（下一帧目标更新，不排队）。"
                        "采样周期用 --point-dt")
    p.add_argument("--queue-high", type=int, default=0,
                   help="高低水位模式（>0 开启，覆盖 --inflight/--seg-time 的交付逻辑）：控制器里最多维持这么多点")
    p.add_argument("--queue-low", type=int, default=3,
                   help="高低水位模式：控制器里剩到这么多点(含)时，把本地攒的下一批整批送入，补到高水位")
    p.add_argument("--refill-max", type=int, default=4,
                   help="高低水位模式：每次补货最多送本地队列最新的几个点，更旧的丢弃。"
                        "越小越跟手（臂直奔手现在的位置），越大越贴合手走过的路径")
    p.add_argument("--point-dt", type=float, default=0.08,
                   help="高低水位模式：路径点的时间间隔 s（配合死区）。控制器吞吐约 12 点/s，低于 0.08 会积压")
    p.add_argument("--inflight", type=int, default=2,
                   help="控制器里最多同时在飞多少个**点**。超过说明臂落后了：这批只留最后一个点、"
                        "并按积压比例提速追赶。越大越顺但允许的滞后越多")
    # 坐标映射
    p.add_argument("--yaw", type=float, default=-90.0, help="头显系绕 Z 转多少度喂给臂")
    p.add_argument("--mirror", choices=("none", "lr", "fb"), default="none")
    p.add_argument("--scale", type=float, default=0.5, help="手动 1m 末端动 scale m")
    p.add_argument("--rot-mode", choices=("none", "roll", "full"), default="roll",
                   help="none=不跟姿态；roll=只跟手掌拧转；full=完整姿态(腕部±50°很容易够不到)")
    p.add_argument("--rot-scale", type=float, default=1.0)
    p.add_argument("--lead-time", type=float, default=0.0,
                   help="用卡尔曼估的手速把目标沿运动方向提前这么多秒：目标 = 滤波位置 + 手速×lead。"
                        "手匀速时臂到达目标的时刻手也刚到，感知延迟被抵消；手急停时会多走 手速×lead 再回来。"
                        "设成实测端到端延迟的 60~80%%。0=关")
    p.add_argument("--lead-max-mm", type=float, default=40.0, help="预测提前量的上限（末端等效 mm）")
    # 滤波
    p.add_argument("--kf-sigma-a", type=float, default=3.0, help="卡尔曼过程噪声：人手加速度量级 m/s²")
    p.add_argument("--kf-sigma-m", type=float, default=0.003, help="卡尔曼量测噪声：头显位置噪声 m")
    p.add_argument("--lp-cutoff", type=float, default=1.5, help="One-Euro 最小截止频率 Hz，0=关")
    p.add_argument("--lp-beta", type=float, default=1.0, help="One-Euro 速度系数")
    p.add_argument("--hand-speed-max", type=float, default=1.0, help="跳变门 m/s，0=关")
    # 死区 / 段速
    p.add_argument("--deadband-mm", type=float, default=3.0, help="入队死区（滤噪）：相对队尾/上一个已发点")
    p.add_argument("--min-seg-mm", type=float, default=10.0,
                   help="交付门槛（段长）：本地队列折线累计不到这个长度先不交付、继续攒。"
                        "MoveL 短段在加速阶段就结束(实测 3mm 段有效速度约 40mm/s)，段长要够臂提速。"
                        "手停下(手速<--stop-speed)时例外，立刻交付最后一点保证到位")
    p.add_argument("--stop-speed", type=float, default=30.0, help="低于此手速(mm/s,末端等效)视为手停了")
    p.add_argument("--seg-time", type=float, default=0.10,
                   help="每段至少覆盖多少秒的手部运动：交付门槛取 max(min-seg-mm, 手速×seg-time)。"
                        "控制器执行一个点有约 80ms 固定开销，段的时长得比它长臂才跟得上")
    p.add_argument("--speed-gain", type=float, default=1.1,
                   help="段速 = 段长/手产生该段的时间 × 该增益。略大于 1 让臂不落后，太大又会冲完干等")
    p.add_argument("--rot-deadband-deg", type=float, default=3.0, help="姿态死区")
    p.add_argument("--speed-min", type=float, default=80.0,
                   help="段速下限 mm/s。太低臂爬着走（实测 40 时一段要 250ms+）；太高臂走完干等下一个点、"
                        "队列空停车再起步。80~120 之间调")
    p.add_argument("--speed-max", type=float, default=500.0, help="段速上限 mm/s")
    p.add_argument("--consume-margin", type=float, default=2.0,
                   help="位置兜底判「点已走完」的余量 mm：TCP 进到路点 (转弯区×1.2 + 该值) 内算完成。"
                        "越大越早放行下一批(在飞数被低估、更连续、延迟略增)，越小越保守")
    p.add_argument("--max-seg-mm", type=float, default=15.0,
                   help="单段最大长度 mm，超过就截断，剩下的下一段接着走。**这是延迟的主旋钮**："
                        "实测执行时间 10mm 段 91ms、25~50mm 段 421ms、50mm+ 段 545ms。"
                        "不限长时队列满丢点→手继续动→下次发的是长段→执行更久，自我恶化。0=不限")
    p.add_argument("--no-elbow", action="store_true",
                   help="不给控制器指定肘角/构型，让它自己解冗余。默认是指定的（每次发命令前刷新成当前实际值）。"
                        "如果仍然慢或频繁 -50002，试这个")
    p.add_argument("--cmd-speed", type=float, default=600.0,
                   help="所有段固定用这个设定速度 mm/s。**设定速度同时决定关节速度档位和加速度**"
                        "（实测 10mm 段 设100=240ms / 设600=162ms；80mm 段 906 vs 364），"
                        "所以一律给高值，段长自会限制实际速度。0=退回按手速算段速")
    p.add_argument("--catchup", type=float, default=0.5,
                   help="追赶系数：控制器里每多积压 1 个点，这批段速提高该比例。"
                        "延迟 ≈ 在飞点数 × 每点执行时间，臂不比手快积压就消不掉。0=不追赶")
    p.add_argument("--zone-mm", type=float, default=100.0,
                   help="转弯区。控制器按档映射成衔接百分比：<1=停车, 1~20=10%%, 20~60=30%%, >60=100%%。"
                        "遥操要连续就给 >60（默认 100），与段长无关")
    p.add_argument("--ws-margin-mm", type=float, default=60.0,
                   help="工作空间盒 x/y 各面向内收这么多 mm。盒子边角臂到不了，撞上就是 -50002 停车重锚")
    p.add_argument("--ws-margin-z-mm", type=float, default=20.0,
                   help="工作空间盒 z 两面向内收这么多 mm。2026-09-21 用 60 时 z 下限抬到 140mm，24%% 时间贴在下限上「下不去」")
    p.add_argument("--accel", type=float, default=1.5,
                   help="控制器加/减速度 = 系统预设 × 该百分比，范围 0.2~1.5。短段全程在加减速，这是缩短每点时间的旋钮。0=不改")
    p.add_argument("--jerk", type=float, default=2.0,
                   help="控制器加加速度 = 系统预设 × 该百分比，范围 0.1~2。越大起停越硬")
    p.add_argument("--max-stale", type=float, default=0.25, help="多久没新帧就摘离合")
    # 臂
    p.add_argument("--robot-ip", default="192.168.2.160")
    p.add_argument("--local-ip", default="192.168.2.222")
    p.add_argument("--no-home", action="store_true", help="启动时不回起始姿态")
    # 手
    p.add_argument("--no-hand", action="store_true")
    p.add_argument("--hand-port", default="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0")
    p.add_argument("--hand-hz", type=float, default=40)
    p.add_argument("--hand-force", type=int, default=400)
    p.add_argument("--hand-speed", type=int, default=1000)
    p.add_argument("--smooth", type=float, default=0.20, help="手指角度一阶低通系数")
    p.add_argument("--pinch-gate", type=float, default=0.45)
    p.add_argument("--freeze-thumb-rot", action="store_true")
    p.add_argument("--calib-file", default=str(DEFAULT_CALIB))
    p.add_argument("--print-hz", type=float, default=5)
    p.add_argument("--trace", default="", help="CSV：时间/原始/滤波/目标/实测/下发（每帧，很大，一般不用）")
    p.add_argument("--point-csv", default="auto",
                   help="逐点全链路时间戳 CSV：auto=logs/points_<日期时间>.csv；none=不写。"
                        "每行一个采样点，列出它在 头显帧/滤波/入队/下发/走完 五个环节的时刻")
    p.add_argument("--log", default="auto",
                   help="运行日志路径。auto=logs/movel_<日期时间>.log；none=不写。发给别人看就发这个文件")
    return p.parse_args()


def e2e_report(ts, targets, actuals, max_shift_s: float = 4.0) -> str:
    """端到端延迟：把实测 TCP 曲线在时间上往前平移 τ，找让它和目标曲线最贴合的 τ。

    这是「手到了某个位置，臂多久之后到同一位置」的直接度量，包含滤波、攒段、交付、
    控制器队列和执行全部环节。要求这一场里手真的动过；只有静止段时结果无意义。
    """
    if len(ts) < 100:
        return "  端到端延迟: 样本不足（接合时长不够）"
    t = np.asarray(ts); T = np.asarray(targets); A = np.asarray(actuals)
    if np.ptp(T, axis=0).max() < 0.03:
        return "  端到端延迟: 目标几乎没动（<3cm），测不出"
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], dt)
    Ti = np.stack([np.interp(grid, t, T[:, i]) for i in range(3)], 1)
    Ai = np.stack([np.interp(grid, t, A[:, i]) for i in range(3)], 1)
    best, best_err, errs = 0.0, np.inf, []
    max_k = int(min(max_shift_s / dt, len(grid) // 2))
    for k in range(0, max_k):
        # 实测提前 k 步 与 目标 对齐：actual(t+τ) ≈ target(t)
        e = np.linalg.norm(Ai[k:] - Ti[:len(grid) - k], axis=1)
        m = float(np.mean(e)); errs.append(m)
        if m < best_err:
            best_err, best = m, k * dt
    e0 = errs[0]
    return (f"  端到端延迟(目标→实测最佳对齐): {best*1000:.0f} ms   "
            f"对齐后平均残差 {best_err*1000:.1f} mm（不平移时 {e0*1000:.1f} mm）  样本 {len(grid)} @ {1/dt:.0f}Hz")


# ──────────────────────────────────────────────────────────────────────────────
def setup_hand(args, calib):
    if args.no_hand:
        return None
    hand = ih.InspireHand6(port=args.hand_port)
    hand.connect()
    names = list(getattr(ih, "DOF_NAMES", ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]))
    for i in range(ih.NUM_DOF):
        if names[i] in ("index", "thumb_bend"):
            calib.limits[i] = hand.limits[i] = (0, 1000)
        else:
            lo, hi = calib.limits.get(i, ih.DEFAULT_LIMITS[i])
            hlo, hhi = hand.limits[i]
            hand.limits[i] = (max(lo, hlo), min(hi, hhi))
    ok, why = hand.liveness()
    if not ok:
        raise RuntimeError(f"灵巧手未就绪: {why}")
    hand.set_force(args.hand_force)
    hand.set_speed(args.hand_speed)
    hand.start_writer()
    print(f"灵巧手已连接 {args.hand_port}  力控 {args.hand_force}g  速度 {args.hand_speed}")
    return hand


def main():
    args = parse_args()
    period = 1.0 / args.hz
    sample_period = 1.0 / max(args.sample_hz, 1e-3)
    flush_period = 1.0 / max(args.flush_hz, 1e-3)

    runlog = None
    if args.log and args.log != "none":
        path = (HERE / "logs" / f"movel_{time.strftime('%Y%m%d_%H%M%S')}.log") if args.log == "auto" else Path(args.log)
        runlog = RunLog(path)
        runlog.line("ARGS", " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))
        print(f"运行日志 → {runlog.path}", flush=True)

    # ── 臂：NRT 直连，不开 RT ──
    cfg = AR5Config(mock=False, real_hand_in_mock=False)
    cfg.use_hand = False
    cfg.arm.use_realtime = False
    cfg.arm.ip, cfg.arm.local_ip = args.robot_ip, args.local_ip
    cfg.cameras.sources = {}
    env = AR5Env(cfg)
    print("连接 AR5（NRT，不开 RT 流）...", flush=True)
    env.connect()
    robot, sdk = env.arm._robot, env.arm.sdk
    margin = np.array([args.ws_margin_mm, args.ws_margin_mm, args.ws_margin_z_mm]) / 1000.0
    ws_lo = np.asarray(cfg.arm.workspace_min, dtype=float) + margin
    ws_hi = np.asarray(cfg.arm.workspace_max, dtype=float) - margin
    if np.any(ws_hi <= ws_lo):
        raise SystemExit(f"--ws-margin-mm {args.ws_margin_mm} 太大，工作空间盒收没了")

    if not args.no_home:
        print("  ⚠ 即将关节回位（MoveAbsJ），3 秒内 Ctrl+C 可取消 ...", flush=True)
        time.sleep(3.0)
        print("  回位中 ...", end=" ", flush=True)
        print("完成" if env.home_joints() is not False else "未在超时内到位")

    direct = args.direct
    watermark = args.queue_high > 0 and not direct
    if direct:
        sample_period = max(args.point_dt, period)
    if watermark:
        args.queue_low = max(1, min(args.queue_low, args.queue_high - 1))
        sample_period = max(args.point_dt, period)
    q = MoveLQueue(robot, sdk, zone_mm=args.zone_mm, vmin=args.speed_min, vmax=args.speed_max,
                   inflight=(args.queue_high if (watermark or direct) else args.inflight),
                   catchup=args.catchup,
                   consume_margin_mm=args.consume_margin)
    if watermark:
        q.catchup_base = args.queue_low
    q.runlog = runlog
    if args.accel > 0:
        q.accel_scale, q.jerk_scale = args.accel, args.jerk
    q.cmd_speed = max(0.0, args.cmd_speed)
    q.send_elbow = not args.no_elbow
    q.max_seg = max(0.0, args.max_seg_mm) / 1000.0
    q.configure()
    print(f"  工作空间(已收边): x[{ws_lo[0]:.2f},{ws_hi[0]:.2f}] y[{ws_lo[1]:.2f},{ws_hi[1]:.2f}] z[{ws_lo[2]:.2f},{ws_hi[2]:.2f}] m")
    if runlog:
        try:
            info = robot.robotInfo({})
            runlog.line("INIT", f"robot={getattr(info,'type','?')} xcore={getattr(info,'version','?')} "
                                f"power={q.power_state()} mode={'watermark' if watermark else 'inflight'} "
                                f"tcp0={np.round(q.last_sent[0]*1000,1).tolist()}mm")
        except Exception as e:  # noqa: BLE001
            runlog.line("INIT", f"robotInfo 读取失败: {e}")
    print("  负载: 沿用控制器已存设定（NRT 路径不写 toolset 负载，2026-09-20 实测写了会掉电）")

    # ── 头显 / 手 ──
    streamer = connect_streamer(args)
    if streamer is None:
        env.close()
        return 1
    calib = Calibration.load(Path(args.calib_file)) if Path(args.calib_file).exists() else Calibration()
    hand = tracker = None
    try:
        hand = setup_hand(args, calib)
    except Exception as e:  # noqa: BLE001
        print(f"  手连不上（{type(e).__name__}: {e}），只跑臂")
        hand = None
    if hand is not None:
        retarget = Retargeter(calib, smooth=args.smooth, pinch_gate=args.pinch_gate)
        tracker = HandTracker(streamer, retarget, hand, args.hand_hz, args.freeze_thumb_rot, False)
        tracker.start()

    # ── 滤波器 ──
    A = yaw_matrix(args.yaw) @ mirror_matrix(args.mirror)
    gate = JumpGate(args.hand_speed_max, period)
    kf = KalmanCV(args.kf_sigma_a, args.kf_sigma_m)
    lp_pos = OneEuro(args.lp_cutoff, args.lp_beta) if args.lp_cutoff > 0 else None
    lp_rot = OneEuro(args.lp_cutoff, args.lp_beta) if args.lp_cutoff > 0 else None

    print("\n" + "─" * 66)
    if direct:
        print(f"  臂 {args.robot_ip}  MoveL 直通（无本地队列）  点间隔 {args.point_dt*1000:.0f}ms  "
              f"在飞≥{args.queue_high} 就丢点")
    elif watermark:
        print(f"  臂 {args.robot_ip}  MoveL 多路点·高低水位  点间隔 {args.point_dt*1000:.0f}ms → 本地队列 → "
              f"控制器剩≤{args.queue_low}点时补到{args.queue_high}点")
    else:
        print(f"  臂 {args.robot_ip}  MoveL 多路点  采样 {args.sample_hz:.0f}Hz → 本地队列 → "
              f"{args.flush_hz:.0f}Hz 列表交付  控制器在飞上限 {args.inflight} 点")
    print(f"  滤波: 跳变门 {args.hand_speed_max}m/s → 卡尔曼(σa={args.kf_sigma_a}, σm={args.kf_sigma_m*1000:.0f}mm)"
          f" → One-Euro({args.lp_cutoff}Hz, β={args.lp_beta})")
    print(f"  死区 {args.deadband_mm}mm / {args.rot_deadband_deg}°  段速 手速×{args.speed_gain}"
          f" ∈ [{args.speed_min:.0f},{args.speed_max:.0f}] mm/s  zone {args.zone_mm}mm")
    print(f"  缩放 x{args.scale}  yaw {args.yaw}°  镜像 {args.mirror}  姿态 {args.rot_mode}")
    print("  ENTER 挂/摘离合   右手保持捏合驱动臂   [ ] 死区   h 停下并回位   q 退出")
    print("─" * 66, flush=True)

    trace = None
    if args.trace:
        trace = open(args.trace, "w")
        trace.write("t,raw_x,raw_y,raw_z,filt_x,filt_y,filt_z,tgt_x,tgt_y,tgt_z,act_x,act_y,act_z,hand_v,sent\n")

    interactive = sys.stdin.isatty()
    old_term = termios.tcgetattr(sys.stdin) if interactive else None
    if interactive:
        tty.setcbreak(sys.stdin.fileno())

    armed = deadman = False
    ref = None                      # (hand_p0, hand_R0, arm_p0, arm_R0)
    prev_frame = None
    last_fresh = time.monotonic()
    last_sample = last_flush = last_print = last_snap = 0.0
    pending = []                    # 本地路径队列：[(trans, rpy, 手速mm/s, 时间戳)]
    # 端到端延迟测量：每 20ms 记一对 (目标位置, 实测 TCP)，退出时找让两条曲线最对齐的时间偏移
    e2e_t, e2e_target, e2e_actual = [], [], []
    last_e2e = 0.0
    t_filt = 0.0
    ptrace = PointTrace() if args.point_csv != "none" else None
    if ptrace is not None:
        q.trace = ptrace
    hand_v = np.zeros(3)
    filt_p = None
    filt_R = None
    target_p = None
    target_rpy = None
    n_frames = n_dead = n_queued = n_lead_capped = 0
    status = ""
    try:
        while True:
            t0 = time.monotonic()
            code = q.poll()
            if code:
                if runlog:
                    runlog.line("ERR", f"exec error {code}: {q.last_error[:300]}")
                print(f"\n[MoveL] 控制器报执行错误 {code}: {q.last_error[:160]}")
                print("        目标可能超出可达范围。停下清队列并重新锚定，手回到可达区再捏合。")
                q.stop()
                ref = None

            # 键盘
            if interactive and select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "q":
                    break
                if ch in ("\r", "\n"):
                    armed = not armed
                    ref = None
                    if not armed:
                        q.stop()
                    if runlog:
                        runlog.line("CLUTCH", "armed" if armed else "disarmed + moveReset")
                    print(f"\n[离合] {'挂上' if armed else '摘下，队列已清'}")
                elif ch == "h":
                    armed, ref = False, None
                    q.stop()
                    print("\n[回位] 关节空间回起始姿态 ...", end=" ", flush=True)
                    try:
                        print("完成" if env.home_joints() is not False else "超时")
                    except Exception as e:  # noqa: BLE001
                        print(f"失败: {e}")
                    q.configure()
                elif ch in ("[", "]"):
                    args.deadband_mm = max(0.5, args.deadband_mm * (1.25 if ch == "]" else 0.8))
                    print(f"\n[死区] {args.deadband_mm:.1f} mm")

            frame = streamer.latest
            fresh = frame is not None and frame is not prev_frame
            data = streamer.get_latest()
            if fresh:
                prev_frame, last_fresh = frame, t0
            stale = (t0 - last_fresh) > args.max_stale
            if stale and armed:
                print(f"\n[链路] 超过 {args.max_stale}s 没有新帧，摘离合")
                if runlog:
                    runlog.line("ERR", f"headset stale >{args.max_stale}s → disarm")
                armed, ref = False, None
                q.stop()
            if data is None:
                time.sleep(period)
                continue

            # 右手捏合 = dead-man
            r_pinch = float(data["right_pinch_distance"])
            deadman = r_pinch < (PINCH_OFF if deadman else PINCH_ON)

            # ── 滤波链（只在新帧上推进） ──
            if fresh:
                n_frames += 1
                raw_p = wrist_xyz(data, "left")
                raw_R = wrist_R(data, "left")
                p_g, ok = gate(raw_p, t0)
                p_k, hand_v = kf(p_g, t0)
                t_filt = t0
                if lp_pos is not None:
                    filt_p = lp_pos(p_k, t0)
                    filt_R = filter_rotation(lp_rot, raw_R, t0) if ok or filt_R is None else filt_R
                else:
                    filt_p, filt_R = p_k, raw_R

            engaged = armed and deadman and not stale and filt_p is not None
            if not engaged:
                ref = None
            elif ref is None:
                if not q.is_idle():
                    status = "等臂停稳再锚定"
                else:
                    arm_p0, arm_rpy0, _ = q.pose()
                    ref = (filt_p.copy(), filt_R.copy(), arm_p0.copy(), rpy_matrix(arm_rpy0))
                    q.last_sent = (arm_p0.copy(), list(arm_rpy0))
                    if runlog:
                        runlog.line("CLUTCH", f"engaged: arm0={np.round(arm_p0*1000,1).tolist()}mm")
                    print("\n[接合] 已锁定基准")

            # ── 绝对目标 ──
            if engaged and ref is not None:
                h0, R0, a0, Ra0 = ref
                hand_rel = filt_p - h0
                if args.lead_time > 0:
                    # 预测提前：沿手速方向提前 lead_time 秒，限幅在 lead_max_mm（末端等效）
                    lead = hand_v * args.lead_time                        # 头显系，米
                    lead_mm = float(np.linalg.norm(lead)) * args.scale * 1000
                    if lead_mm > args.lead_max_mm:
                        lead *= args.lead_max_mm / lead_mm
                    hand_rel = hand_rel + lead
                    n_lead_capped += int(lead_mm > args.lead_max_mm)
                target_p = a0 + A @ hand_rel * args.scale
                target_p = np.clip(target_p, ws_lo, ws_hi)
                if args.rot_mode == "none":
                    R_t = Ra0
                else:
                    rel = A @ (filt_R @ R0.T) @ A.T
                    rot = matrix_to_axis_angle(rel) * args.rot_scale
                    if args.rot_mode == "roll":
                        axis = Ra0[:, 2]
                        rot = axis * float(rot @ axis)
                    R_t = axis_angle_to_matrix(rot) @ Ra0
                target_rpy = list(matrix_to_rpy(R_t))

                # ── 采样：按周期把滤波后的目标放进本地路径队列（死区相对队尾/上一个已发点） ──
                if t0 - last_sample >= sample_period:
                    last_sample = t0
                    lp, lrpy = (pending[-1][0], pending[-1][1]) if pending else q.last_sent
                    if not pending and q.last_sent_t is None:
                        q.last_sent_t = t0          # 锚定/清队列后的第一段：时间从现在起算
                    dpos = float(np.linalg.norm(target_p - lp)) * 1000
                    drot = np.rad2deg(rotation_distance(target_rpy, lrpy))
                    if dpos >= args.deadband_mm or drot >= args.rot_deadband_deg:
                        v_mm = float(np.linalg.norm(hand_v)) * args.scale * 1000
                        pid = 0
                        if ptrace is not None:
                            pid = ptrace.new(t_frame=last_fresh, t_filt=t_filt, t_queue=t0,
                                             x=float(target_p[0]), y=float(target_p[1]), z=float(target_p[2]),
                                             hand_mm_s=float(np.linalg.norm(hand_v)) * args.scale * 1000)
                        pending.append((target_p.copy(), list(target_rpy), v_mm, t0, pid))
                        n_queued += 1
                        status = f"队列{len(pending)}"
                    else:
                        n_dead += 1
                        status = "死区"
                # ── 交付（直通）：没有本地队列，过了死区的点立刻单条 moveAppend ──
                # 唯一的闸是控制器在飞上限，满了就丢掉这个点（下一帧的目标更新）。
                if direct and pending:
                    n_fly = q.inflight()
                    if n_fly >= args.queue_high:
                        if ptrace is not None:
                            for d in pending:
                                ptrace.set(d[4], fate="dropped_full")
                        q.stats["held_full"] += 1
                        status = f"满{n_fly}丢"
                    else:
                        status = q.push_batch(pending[-1:], sample_period, args.speed_gain)
                        if ptrace is not None:
                            # push_batch 自己也有一道在飞闸；被它拒掉的点要标出来，
                            # 否则 CSV 里留成 queued，看着像滞留其实是被静默丢弃。
                            rejected = status.startswith("满")
                            for d in pending[:-1]:
                                ptrace.set(d[4], fate="dropped_old")
                            if rejected:
                                ptrace.set(pending[-1][4], fate="dropped_full")
                    pending.clear()
                # ── 交付（高低水位）：控制器剩 ≤ low 个点时，把本地攒的整批送入补到 high ──
                elif watermark and pending and t0 - last_flush >= flush_period:
                    last_flush = t0
                    n_fly = q.inflight()
                    hand_mm_s = float(np.linalg.norm(hand_v)) * args.scale * 1000
                    if n_fly <= args.queue_low or (hand_mm_s < args.stop_speed and n_fly < args.queue_high):
                        room = max(1, min(args.queue_high - n_fly, args.refill_max))
                        batch = pending
                        if len(batch) > room:
                            # 只送最新的 room 个点：臂要去的是手现在的位置，不是重放手走过的路。
                            # 2026-09-21 实测送旧路径时臂在重放几秒前的轨迹。
                            q.stats["thinned"] += len(batch) - room
                            if ptrace is not None:
                                for d in batch[:-room]:
                                    ptrace.set(d[4], fate="dropped_old")
                            batch = batch[-room:]
                        status = q.push_batch(batch, sample_period, args.speed_gain)
                        pending.clear() if not status.startswith("满") else None
                    else:
                        status = f"水位{n_fly}>{args.queue_low} 攒{len(pending)}"
                # ── 交付（默认）：本地队列整批以列表 MoveL 交给控制器 ──
                elif pending and t0 - last_flush >= flush_period:
                    last_flush = t0
                    prev = q.last_sent[0]
                    path_mm = 0.0
                    for tp_, _, _, _, _ in pending:
                        path_mm += float(np.linalg.norm(tp_ - prev)) * 1000
                        prev = tp_
                    hand_mm_s = float(np.linalg.norm(hand_v)) * args.scale * 1000
                    # 交付门槛：每段至少 min-seg-mm，且至少覆盖 seg-time 秒的手部运动
                    need_mm = max(args.min_seg_mm, hand_mm_s * args.seg_time)
                    if path_mm < need_mm and hand_mm_s >= args.stop_speed:
                        q.stats["held_short"] += 1
                        status = f"攒{len(pending)}点{path_mm:.0f}/{need_mm:.0f}mm"
                    else:
                        status = q.push_batch(pending, sample_period, args.speed_gain)
                        if not status.startswith("满"):
                            pending.clear()
                        else:
                            pending = pending[-1:]        # 队列满：只留最新点等下一次
            elif not engaged:
                if pending and ptrace is not None:
                    for d in pending:
                        ptrace.set(d[4], fate="dropped_disengage")
                pending.clear()
                status = "未接合" if armed else "未挂离合·按ENTER"

            if engaged and ref is not None and target_p is not None and t0 - last_e2e >= 0.02:
                last_e2e = t0
                ap_, _, _ = q.pose()
                e2e_t.append(t0); e2e_target.append(target_p.copy()); e2e_actual.append(ap_)

            if trace is not None and filt_p is not None:
                ap, _, _ = q.pose()
                tp = target_p if target_p is not None else np.full(3, np.nan)
                trace.write(",".join(f"{v:.5f}" for v in [t0, *wrist_xyz(data, 'left'), *filt_p, *tp, *ap,
                                                          float(np.linalg.norm(hand_v))]) + f",{status}\n")

            if runlog and t0 - last_snap >= 1.0:
                last_snap = t0
                ap_, _, _ = q.pose()
                lag_ = (np.linalg.norm(target_p - ap_) * 1000) if target_p is not None else 0.0
                prod_, cons_, wait_, per_ = q.rates(2.0)
                s_ = q.stats
                runlog.line("SNAP", f"engaged={int(bool(engaged and ref is not None))} 落后={lag_:.0f}mm "
                            f"手速={np.linalg.norm(hand_v)*1000:.0f}mm/s 产={prod_:.1f}/s 耗={cons_:.1f}/s "
                            f"等待={wait_:.0f}ms 每点={per_:.0f}ms 队={len(pending)} 飞={q.inflight()} "
                            f"发={s_['appended']} 入队={n_queued} 死区={n_dead} 满扣={s_['held_full']} "
                            f"续攒={s_['held_short']} 抽稀={s_['thinned']} 事件={s_['events']} 兜底={s_['pose_marks']} "
                            f"tcp={np.round(ap_*1000).astype(int).tolist()} status={status}")

            if args.print_hz > 0 and t0 - last_print >= 1.0 / args.print_hz:
                last_print = t0
                ap, _, _ = q.pose()
                lag = (np.linalg.norm(target_p - ap) * 1000) if target_p is not None else 0.0
                tag = "接合" if engaged and ref is not None else ("等右手捏合" if armed else "未挂离合")
                s = q.stats
                prod, cons, wait_ms, per_pt = q.rates(2.0)
                print(f"\r[{tag:^10}] R捏={r_pinch:.3f} 末端({ap[0]:+.3f},{ap[1]:+.3f},{ap[2]:+.3f}) "
                      f"落后 {lag:5.1f}mm 手速 {np.linalg.norm(hand_v)*1000:4.0f}mm/s "
                      f"| 产 {prod:4.1f}/s 耗 {cons:4.1f}/s 等待 {wait_ms:4.0f}ms 每点 {per_pt:3.0f}ms "
                      f"| 队{len(pending)} 飞{q.inflight()} 发{s['appended']} {status:<14}", end="", flush=True)

            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        if old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        print("\n收尾 ...")
        try:
            q.stop()
        except Exception as e:  # noqa: BLE001
            print(f"  清队列失败: {e}")
        if tracker is not None:
            tracker.stop()
            tracker.join(timeout=1.0)
        if hand is not None:
            try:
                hand.stop_writer()
                hand.close()
            except Exception:  # noqa: BLE001
                pass
        if trace is not None:
            trace.close()
        s = q.stats
        n = max(1, s["appended"])
        print(f"  头显帧 {n_frames}  跳变门拒 {gate.rejected} 帧(最大 {gate.max_mm:.0f}mm)"
              + (f"  预测提前 {args.lead_time*1000:.0f}ms(限幅触发 {n_lead_capped} 次)" if args.lead_time > 0 else ""))
        print(f"  采样点: 死区挡 {n_dead}  进本地队列 {n_queued}  实际发出 {s['appended']} 点 / {s['batches']} 批"
              f"(最大 {s['batch_max']} 点)  落后抽稀丢 {s['thinned']} 点")
        print(f"  平均段长 {s['seg_len_sum']/n:.1f}mm  平均段速 {s['speed_sum']/n:.0f}mm/s"
              f"  单段截断 {s['seg_clipped']} 次")
        print(f"  控制器在飞峰值 {s['max_inflight_seen']} 点  队列满而暂扣 {s['held_full']} 次  "
              f"段太短而续攒 {s['held_short']} 次")
        print(f"  moveStart {s['starts']}  已在动 {s['already_moving']}  队列空 {s['queue_empty']}"
              f"  执行错误 {s['exec_errors']}")
        print(f"  消费判据: 事件回调{'已注册' if q.watcher_ok else '未注册'}  解出 {s['events']} 条  "
              f"位置兜底标记 {s['pose_marks']} 次"
              + ("   ⚠ 事件一条没解出来，看日志里的 [EVT] 原始字典" if s["events"] == 0 else ""))
        summary_txt = q.summary()
        e2e_txt = e2e_report(e2e_t, e2e_target, e2e_actual)
        print(summary_txt)
        print(e2e_txt)
        if ptrace is not None and ptrace.rows:
            cpath = (HERE / "logs" / f"points_{time.strftime('%Y%m%d_%H%M%S')}.csv") \
                if args.point_csv == "auto" else Path(args.point_csv)
            t_ref = min((r["t_frame"] for r in ptrace.rows.values() if isinstance(r["t_frame"], float)), default=0.0)
            ptrace.dump(cpath, t_ref)
            done_rows = [r for r in ptrace.rows.values() if isinstance(r.get("t_done"), float)]
            print(f"  逐点 CSV: {cpath}  共 {len(ptrace.rows)} 点（走完 {len(done_rows)}）")
            if done_rows:
                stage = lambda a, b: np.median([r[b] - r[a] for r in done_rows  # noqa: E731
                                                if isinstance(r[a], float) and isinstance(r[b], float)]) * 1000
                print(f"  各环节中位耗时 ms: 帧→滤波 {stage('t_frame','t_filt'):.0f} | "
                      f"滤波→入队 {stage('t_filt','t_queue'):.0f} | 入队→下发 {stage('t_queue','t_push'):.0f} | "
                      f"下发→走完 {stage('t_push','t_done'):.0f} | 合计 {stage('t_frame','t_done'):.0f}")
        if runlog:
            runlog.line("STAT", f"frames={n_frames} gate_rej={gate.rejected} dead={n_dead} queued={n_queued} "
                                f"lead_capped={n_lead_capped} stats={ {k: (round(v,1) if isinstance(v,float) else v) for k,v in s.items()} }")
            for ln in (summary_txt + "\n" + e2e_txt).splitlines():
                runlog.line("STAT", ln.strip())
            runlog.close()
            print(f"运行日志已写到 {runlog.path}")
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
