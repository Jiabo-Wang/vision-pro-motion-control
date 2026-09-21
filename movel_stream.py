#!/usr/bin/env python3
"""Batch MoveL transport for absolute TCP targets in the robot base frame.

submit() is local; flush() performs SDK calls. This is a queued NRT backend:
batching does not provide an end-to-end latency guarantee or remove SDK blocking.
Position AND orientation deadbands suppress unchanged targets. Accepted targets
still use the SDK list overload; the newest subthreshold target is not forced in.
"""
from __future__ import annotations

import time
import numpy as np
from pose_math import rpy_matrix, rotation_distance


class MoveLStreamer:
    LADDER = (0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

    def __init__(self, robot, sdk, *, speed=200.0, zone=3.0, lead_ms=100.0,
                 min_step_mm=0.5, cache=300, buffer_max=64,
                 min_rotation_deg=0.5, max_rotation_lead=0.15,
                 tcp_z=0.0, payload=0.0, payload_com_z=0.0):
        vals = [speed, zone, lead_ms, min_step_mm, min_rotation_deg,
                max_rotation_lead, tcp_z, payload, payload_com_z]
        if not np.isfinite(vals).all() or speed <= 0 or lead_ms <= 0:
            raise ValueError('invalid MoveL limits')
        if min_step_mm < 0 or min_rotation_deg < 0 or max_rotation_lead <= 0:
            raise ValueError('deadbands must be nonnegative; rotation lead must be positive')
        if not 1 <= buffer_max <= 100 or not 1 <= cache <= 1000:
            raise ValueError('buffer_max must be 1..100, SDK cache 1..1000')
        self.robot, self.sdk = robot, sdk
        self.speed, self.zone = float(speed), float(zone)
        self.max_lead = speed * lead_ms / 1e6
        self.min_step = min_step_mm / 1000.0
        self.min_rotation = np.deg2rad(min_rotation_deg)
        self.max_rotation_lead = max_rotation_lead
        self.tcp_z, self.payload, self.payload_com_z = tcp_z, payload, payload_com_z
        self.cache, self.buffer_max = int(cache), int(buffer_max)
        self._last_target = self._last_rpy = None
        self._elbow, self._conf = 0.0, None
        self._running = self._pending = False
        self._buf, self._buf_anchor = [], None
        self.batch_append = True
        self.adaptive_speed, self.min_speed, self.speed_gain = False, 20.0, 1.3
        self._v_hand = 0.0
        self._last_sub = self._last_sub_t = None
        self._seg_speed = self.speed
        self._idle_t0 = self._last_tcp = self._last_tcp_t = None
        self.obs_speed = self._idle_acc = self._busy_acc = 0.0
        self._stalls = []
        self.stats = dict.fromkeys(('samples', 'appended', 'skipped_lead', 'skipped_tiny',
            'restarts', 'already_moving', 'queue_empty', 'start_failed', 'errors',
            'buffered', 'dropped_overflow', 'decimated', 'flushes', 'batch_max',
            'batch_sum', 'over_budget', 'batch_calls', 'batch_fallback', 'append_calls',
            'execution_errors', 'stops'), 0)
        self._w = dict.fromkeys(self.stats, 0)
        self._w_t = time.monotonic()
        self.last_event = {}
        self.last_error = ''

    @staticmethod
    def _check(ec, action):
        if ec.get('ec', 0):
            raise RuntimeError(f'{action}: {ec}')

    def _call(self, method, *args):
        ec = {}
        result = method(*args, ec)
        self._check(ec, getattr(method, '__name__', 'SDK'))
        return result

    def _bump(self, key, n=1):
        self.stats[key] = self.stats.get(key, 0) + n
        self._w[key] = self._w.get(key, 0) + n

    def is_idle(self):
        # An error/unknown/drag state must not be mistaken for permission to start.
        return self._call(self.robot.operationState) == self.sdk.OperationState.idle

    def pose(self):
        cp = self._call(self.robot.cartPosture, self.sdk.CoordinateType.flangeInBase)
        p = np.asarray(cp.trans, dtype=float) + rpy_matrix(cp.rpy) @ [0., 0., self.tcp_z]
        return p, list(cp.rpy), cp

    def tcp(self):
        return self.pose()[0]

    def wait_until_stopped(self, timeout=8.0, log=print):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_idle():
                return True
            time.sleep(0.05)
        log('[MoveL] 等待控制器 idle 超时，取消配置')
        return False

    def configure(self, log=print):
        if not self.wait_until_stopped(log=log):
            return False
        try:
            self._call(self.robot.setMotionControlMode, self.sdk.MotionControlMode.NrtCommandMode)
            self._call(self.robot.setOperateMode, self.sdk.OperateMode.automatic)
            # Power on IMMEDIATELY after the mode switch, exactly like the
            # pre-2026-09-20 code that ran 1395 successful moveStarts. The RT
            # teardown leaves the motors off; in the SDK log a real power-on
            # takes ~214ms to return, while every run that put toolset()/set*
            # calls in between saw setPowerState return in ~41ms as a no-op
            # and the motors stayed off. Order matters here; readback below.
            t_pw = time.monotonic()
            self._call(self.robot.setPowerState, True)
            log(f'[MoveL] setPowerState(True) 耗时 {1000*(time.monotonic()-t_pw):.0f}ms')
            self.ensure_power(log=log)
            # NRT has its own explicit tool/workpiece configuration.
            # setToolset PERSISTS on the controller. Runs earlier today wrote
            # load m=1 / zero inertia into it, and every setToolset carrying that
            # load was followed within ~2s by the motors dropping out with no
            # safety event. So: desired state = tool frames + NO load (the
            # controller never held a load in any working run; RT setLoad was
            # always -28706). Only write when the controller differs, so a
            # healthy controller is not touched at all.
            tool = self._call(self.robot.toolset)
            end, ref = self.sdk.Frame(), self.sdk.Frame()
            end.trans = [0., 0., self.tcp_z]
            end.rpy = ref.rpy = [0., 0., 0.]
            ref.trans = [0., 0., 0.]
            want_mass = 0.0

            def _matches(t):
                return (np.allclose(t.end.trans, end.trans, atol=1e-8)
                        and np.allclose(t.end.rpy, end.rpy, atol=1e-8)
                        and np.allclose(t.ref.trans, ref.trans, atol=1e-8)
                        and np.allclose(t.ref.rpy, ref.rpy, atol=1e-8)
                        and abs(float(getattr(t.load, 'mass', 0.0)) - want_mass) < 1e-9)

            if _matches(tool):
                log('[MoveL] NRT toolset 已是目标值，不重写')
            else:
                log(f'[MoveL] NRT toolset 需要重写：控制器当前 load m={getattr(tool.load, "mass", "?")}'
                    f' end={list(tool.end.trans)}')
                tool.end, tool.ref = end, ref
                tool.load = self.sdk.Load(want_mass, [0., 0., 0.], [0., 0., 0.])
                self._call(self.robot.setToolset, tool)
                actual = self._call(self.robot.toolset)
                if not _matches(actual):
                    raise RuntimeError('NRT toolset readback mismatch')
            self._call(self.robot.setDefaultConfOpt, False)
            self._call(self.robot.setMaxCacheSize, self.cache)
            self._call(self.robot.setDefaultSpeed, self.speed)
            self._call(self.robot.setDefaultZone, self.zone)
            self.ensure_power()
        except Exception as exc:
            self.last_error = str(exc)
            log(f'[MoveL] 配置失败: {exc}')
            return False
        log(f'[MoveL] NRT工具/基座坐标已核对，TCP z={self.tcp_z*1000:.1f}mm')
        return True

    def ensure_power(self, wait_s=3.0, attempts=3, log=print):
        """Motors must be on before moveStart (ec=-17 otherwise). Read back;
        if off, retry (operate mode, then power) a few times with a pause, and
        fail loudly if the controller still stays off."""
        state = self._call(self.robot.powerState)
        if str(state).endswith('.on'):
            return
        for i in range(attempts):
            log(f'[MoveL] 电机 {state}，第 {i+1}/{attempts} 次上电 ...')
            self._call(self.robot.setOperateMode, self.sdk.OperateMode.automatic)
            t_pw = time.monotonic()
            self._call(self.robot.setPowerState, True)
            log(f'[MoveL]   setPowerState 返回耗时 {1000*(time.monotonic()-t_pw):.0f}ms')
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                time.sleep(0.1)
                state = self._call(self.robot.powerState)
                if str(state).endswith('.on'):
                    return
            time.sleep(1.0)
        raise RuntimeError(f'controller stays powered off after setPowerState(True): {state}')

    def start(self, log=print):
        self.ensure_power()
        p, rpy, cp = self.pose()
        self._call(self.robot.moveReset)
        self._elbow, self._conf = cp.elbow, list(cp.confData)
        self._last_target, self._last_rpy = p.copy(), rpy
        self._running, self._pending = True, False
        self._buf, self._buf_anchor = [], None
        self._last_sub = self._last_sub_t = None
        self._v_hand = 0.
        log(f'[MoveL] TCP起点 {np.round(p*1000, 2).tolist()}mm；列表批量提交')

    def suspend(self):
        """Clutch release / stale input: brake and discard pending motion."""
        was_running = self._running
        self._running = self._pending = False
        self._buf, self._buf_anchor = [], None
        if was_running:
            # Brake + discard via moveReset only. robot.stop() powers this AR5
            # off (SDK log 2026-09-20: "stop end (0)" followed by moveStart -17),
            # despite the 0.6.0 header calling it a stop2. Never resume old commands.
            self._call(self.robot.moveReset)
            self._bump('stops')

    def stop(self, go_home=None, log=print):
        self.suspend()
        if go_home is not None:
            raise ValueError('home must be a separate explicit motion after stopping')

    def _changed(self, trans, rpy, anchor, anchor_rpy):
        if anchor is None:
            return True
        dp = float(np.linalg.norm(trans-anchor))
        dr = rotation_distance(rpy, anchor_rpy) if anchor_rpy is not None else 0.
        # Zero thresholds still suppress exact duplicates.
        return dp > max(self.min_step, 1e-9) - 1e-12 or dr > max(self.min_rotation, 1e-7)

    def submit(self, trans, rpy=None, aa=None):
        if not self._running:
            return 'stopped'
        trans = np.asarray(trans, dtype=float).reshape(3).copy()
        rpy = list(self._last_rpy if rpy is None else rpy)
        aa = None if aa is None else np.asarray(aa, dtype=float).reshape(3).copy()
        if not np.isfinite(trans).all() or len(rpy) != 3 or not np.isfinite(rpy).all() or (aa is not None and not np.isfinite(aa).all()):
            raise ValueError('non-finite or malformed MoveL target')
        self._bump('samples')
        now = time.monotonic()
        if self._last_sub is not None:
            dt = now-self._last_sub_t
            if 1e-4 < dt < 1.0:
                self._v_hand = .7*self._v_hand + .3*np.linalg.norm(trans-self._last_sub)/dt*1000
        self._last_sub, self._last_sub_t = trans, now
        self._buf.append((trans, rpy, aa))
        self._bump('buffered')
        if len(self._buf) > self.buffer_max:
            self._buf.pop(0)
            self._bump('dropped_overflow')
        return 'buf'

    def pending_points(self):
        return len(self._buf)

    def _thin_by_spacing(self, pts, anchor):
        keep, prev, prev_rpy = [], anchor, self._last_rpy
        for p in pts:
            if self._changed(p[0], p[1], prev, prev_rpy):
                keep.append(p)
                prev, prev_rpy = p[:2]
        # Do not force a subthreshold latest point, including when keep is empty.
        self._bump('skipped_tiny', len(pts)-len(keep))
        return keep

    def _path_len(self, pts, anchor):
        total, prev = 0., anchor
        for p in pts:
            total += float(np.linalg.norm(p[0]-prev))
            prev = p[0]
        return total

    def _decimate(self, pts, budget, anchor):
        if not pts or self._path_len(pts, anchor) <= budget:
            return pts
        for stride in range(2, len(pts)+1):
            keep = list(reversed(pts[::-1][::stride]))
            if self._path_len(keep, anchor) <= budget:
                self._bump('decimated', len(pts)-len(keep))
                return keep
        self._bump('decimated', len(pts)-1)
        self._bump('over_budget')
        # Caller limits this endpoint to the remaining position/rotation budget.
        return pts[-1:]

    def poll_event(self):
        if not hasattr(self.sdk, 'Event'):
            return
        event = self._call(self.robot.queryEventInfo, self.sdk.Event.moveExecution)
        if not event:
            return
        self.last_event = event
        error = event.get('error', 0)
        if isinstance(error, dict):
            code = error.get('ec', error.get('value', 0))
        elif isinstance(error, (int, float)):
            code = error
        else:
            value = getattr(error, 'value', 0)
            code = value() if callable(value) else value
        if code:
            self.last_error = f'MoveL执行失败: {event}'
            self._bump('execution_errors')
            self.suspend()
            raise RuntimeError(self.last_error)

    def flush(self):
        if not self._running:
            return 'stopped', None
        self.poll_event()
        stopped = self.is_idle()
        cur, cur_rpy, _ = self.pose()
        self._track_stall(cur, stopped)
        if not self._buf:
            if self._pending and stopped:
                return self._try_start(), None
            return 'empty', None
        lead = float(np.linalg.norm(self._last_target-cur))
        rot_lead = rotation_distance(self._last_rpy, cur_rpy)
        budget = self.max_lead-lead
        if (budget <= 1e-8 or rot_lead >= self.max_rotation_lead) and not stopped:
            self._bump('skipped_lead', max(0, len(self._buf)-1))
            self._buf = self._buf[-1:]  # retry latest even if no new frame arrives
            return 'lead', None
        pts = self._thin_by_spacing(self._buf, self._last_target)
        self._buf = []
        if not pts:
            if self._pending and stopped:
                self._try_start()
            return 'tiny', None
        pts = self._decimate(pts, max(budget, 0.), self._last_target)
        # A large target is advanced in bounded steps instead of violating the
        # budget. Keep the latest desired endpoint locally for the next flush.
        from vel_kin import load_vel_module
        tf = load_vel_module('transforms')
        anchor, R = self._last_target.copy(), rpy_matrix(self._last_rpy)
        distance_left = self.max_lead if stopped else max(budget, 0.)
        angle_left = self.max_rotation_lead if stopped else max(self.max_rotation_lead-rot_lead, 0.)
        bounded = []
        for p, rpy, aa in pts:
            d = np.linalg.norm(p-anchor)
            target_R = rpy_matrix(rpy)
            rot = tf.matrix_to_axis_angle(target_R @ R.T)
            angle = np.linalg.norm(rot)
            ratio = min(1., distance_left/max(d, 1e-12), angle_left/max(angle, 1e-12))
            new_p = anchor + ratio*(p-anchor)
            new_R = tf.axis_angle_to_matrix(rot*ratio) @ R
            if ratio > 1e-8:
                bounded.append((new_p, list(tf.matrix_to_rpy(new_R)), tf.matrix_to_axis_angle(new_R)))
            if ratio < 1.-1e-9:
                self._buf = [pts[-1]]
                break
            distance_left -= d
            angle_left -= angle
            anchor, R = p, target_R
        if not bounded:
            self._buf = pts[-1:]
            return 'lead', None
        self._bump('flushes')
        self._bump('batch_sum', len(bounded))
        self.stats['batch_max'] = max(self.stats['batch_max'], len(bounded))
        self._seg_speed = self.segment_speed()
        sent, last6 = self._append_batch(bounded) if self.batch_append else self._append_one_by_one(bounded)
        return (f'批{sent}+'+self._try_start() if stopped else f'批{sent}'), last6

    def _cmd(self, trans, rpy, speed=None):
        target = self.sdk.CartesianPosition()
        target.trans, target.rpy = list(trans), list(rpy)
        if self._conf is not None:
            target.elbow, target.hasElbow, target.confData = self._elbow, True, list(self._conf)
        command = self.sdk.MoveLCommand(target)
        command.speed = self.speed if speed is None else float(speed)
        command.zone = self.zone
        return command

    def _record(self, pts):
        for trans, rpy, aa in pts:
            self._bump('appended')
            self._last_target, self._last_rpy = trans.copy(), list(rpy)
        self._pending = True
        p, _, aa = pts[-1]
        return len(pts), None if aa is None else np.concatenate([p, aa])

    def _append_batch(self, pts):
        cmds = [self._cmd(p, r, self._seg_speed) for p, r, _ in pts]
        try:
            self._call(self.robot.moveAppend, cmds, self.sdk.PyString(''))
        except TypeError:
            self.batch_append = False
            self._bump('batch_fallback')
            return self._append_one_by_one(pts)
        except Exception:
            self._bump('errors')
            raise
        self._bump('batch_calls')
        self._bump('append_calls')
        return self._record(pts)

    def _append_one_by_one(self, pts):
        last6 = None
        for point in pts:
            self._call(self.robot.moveAppend, self._cmd(point[0], point[1], self._seg_speed), self.sdk.PyString(''))
            _, last6 = self._record([point])
            self._bump('append_calls')
        return len(pts), last6

    def _try_start(self):
        ec = {}
        self.robot.moveStart(ec)
        code = ec.get('ec', 0)
        if code == -20:
            self._bump('already_moving')
            # Keep pending: the previous queue may be finishing concurrently.
            return '已在动'
        if code == 768:
            self._pending = False
            self._bump('queue_empty')
            return '队列空'
        if code:
            self._bump('start_failed')
            self._check(ec, 'moveStart')
        self._pending = False
        self._bump('restarts')
        self.note_restart()
        return 'start'

    def update(self, trans, rpy=None, aa=None):
        self.submit(trans, rpy, aa)
        return self.flush()[0]

    def segment_speed(self):
        return float(min(self.speed, max(self.min_speed, self._v_hand*self.speed_gain))) if self.adaptive_speed else self.speed

    @property
    def deadband_mm(self):
        return self.min_step*1000

    def set_deadband(self, mm):
        old = self.deadband_mm
        self.min_step = max(0., float(mm))/1000
        self._w = dict.fromkeys(self.stats, 0)
        self._w_t = time.monotonic()
        return old

    def step_deadband(self, up):
        values = self.LADDER if up else reversed(self.LADDER)
        nxt = next((v for v in values if (v > self.deadband_mm+1e-9 if up else v < self.deadband_mm-1e-9)), self.deadband_mm)
        return self.set_deadband(nxt), nxt

    def _track_stall(self, cur, stopped):
        now = time.monotonic()
        if self._last_tcp_t is not None:
            dt = now-self._last_tcp_t
            if dt > 0:
                self.obs_speed = .8*self.obs_speed + .2*np.linalg.norm(cur-self._last_tcp)/dt*1000
                if stopped:
                    self._idle_acc += dt
                else:
                    self._busy_acc += dt
        self._last_tcp, self._last_tcp_t = cur.copy(), now
        if stopped and self._idle_t0 is None:
            self._idle_t0 = now
        if not stopped:
            self.note_restart()

    def note_restart(self):
        if self._idle_t0 is not None:
            self._stalls.append(time.monotonic()-self._idle_t0)
            self._idle_t0 = None

    def window_batch_mean(self):
        return self._w['batch_sum']/max(1, self._w['flushes'])

    def window_summary(self):
        return f'死区 {self.deadband_mm:.1f}mm / {np.rad2deg(self.min_rotation):.1f}°；每批 {self.window_batch_mean():.1f} 点；过滤 {self._w["skipped_tiny"]}'

    def stall_summary(self):
        total = self._idle_acc+self._busy_acc
        return (f'启动/恢复 {self.stats["restarts"]} 次（含首次）；空闲采样占比 '
                f'{100*self._idle_acc/max(total,1e-9):.0f}%；手速 {self._v_hand:.1f}mm/s，臂速 {self.obs_speed:.1f}mm/s')

    def report(self):
        return (f'MoveL: 采样 {self.stats["samples"]}；过滤 {self.stats["skipped_tiny"]}；'
                f'下发 {self.stats["appended"]} 点 / {self.stats["append_calls"]} 次调用；'
                f'执行错误 {self.stats["execution_errors"]}\n  {self.window_summary()}\n  {self.stall_summary()}')
