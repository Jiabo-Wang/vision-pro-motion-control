#!/usr/bin/env python3
"""采数据:把一条 episode 攒在内存里,结束时后台写盘。

每条 episode 存成一个 `.npz` + 每路相机一个 `.mp4`。

**写盘必须在后台线程上。** servoJ 要求每 8ms 收到一个关节角,主循环停 2 秒就会被
控制器判成通信丢包。实测 600 帧两路相机写一次要 2 秒出头。

字段(13 自由度 = AR5 七轴 + 因时手六维):

    state      13   实测:7 个关节角(rad) + 6 个手指角(0~1000)
    action     13   指令:同上。servoJ 下发的就是绝对关节角,手也是绝对指令角,
                    所以动作空间用绝对值,不用增量
    ee_pose     6   实测末端绝对位姿:位置(m) + 轴角(rad)
    ee_target   6   指令末端位姿。留着以后想换动作空间时重算
    timestamp   n   每帧相对第一帧的秒数,实测。**转数据集用它,不要用标称 fps**
    fps_measured    实测帧率
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

STATE_DIM = 13
ARM_DOF = 7
HAND_DOF = 6


class EpisodeRecorder:
    def __init__(self, out_dir, camera_keys, fps=30.0, task="", codec="mp4v",
                 min_frames=60, min_ee_path_mm=50.0):
        self.out_dir = Path(out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.camera_keys = list(camera_keys)
        self.fps, self.task, self.codec = float(fps), task, codec
        self.min_frames, self.min_ee_path_mm = int(min_frames), float(min_ee_path_mm)
        self._writer = None
        self._write_error = None
        self.reset()

    def reset(self):
        self.state, self.action = [], []
        self.ee_pose, self.ee_target = [], []
        self.stamps = []
        self.frames = {k: [] for k in self.camera_keys}
        self.dropped_frames = 0          # 相机没给新图的次数

    def __len__(self):
        return len(self.state)

    # ── 录一帧 ────────────────────────────────────────────────────────────
    def add(self, q_meas, q_cmd, hand_meas, hand_cmd, ee_pose, ee_target, images):
        """images: {相机名: RGB uint8 图}。缺某一路就跳过这一帧,不留空洞。"""
        for k in self.camera_keys:
            if images is None or images.get(k) is None:
                self.dropped_frames += 1
                return False
        self.stamps.append(time.perf_counter())
        hm = np.zeros(HAND_DOF) if hand_meas is None else np.asarray(hand_meas, dtype=float)
        hc = np.zeros(HAND_DOF) if hand_cmd is None else np.asarray(hand_cmd, dtype=float)
        self.state.append(np.concatenate([np.asarray(q_meas, float)[:ARM_DOF], hm]).astype(np.float32))
        self.action.append(np.concatenate([np.asarray(q_cmd, float)[:ARM_DOF], hc]).astype(np.float32))
        self.ee_pose.append(np.asarray(ee_pose, dtype=np.float32))
        self.ee_target.append(np.asarray(ee_target, dtype=np.float32))
        for k in self.camera_keys:
            self.frames[k].append(images[k])
        return True

    def measured_fps(self):
        if len(self.stamps) < 2:
            return None
        span = self.stamps[-1] - self.stamps[0]
        return (len(self.stamps) - 1) / span if span > 1e-6 else None

    # ── 存之前看一眼这条能不能用 ──────────────────────────────────────────
    def warnings(self):
        n = []
        if len(self) < self.min_frames:
            n.append(f"只有 {len(self)} 帧,短于 {self.min_frames}")
        if len(self) < 2:
            return n
        a = np.stack(self.action)
        hand = a[:, ARM_DOF:]
        if float(np.max(hand.max(0) - hand.min(0))) < 20:
            n.append("手全程几乎没动,手那六维学不到东西")
        p = np.stack(self.ee_pose)[:, :3]
        path = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) * 1000
        if path < self.min_ee_path_mm:
            n.append(f"末端一共只走了 {path:.0f} mm(阈值 {self.min_ee_path_mm:.0f})")
        m = self.measured_fps()
        if m and abs(m - self.fps) > self.fps * 0.15:
            n.append(f"实测帧率 {m:.1f} Hz,标称 {self.fps:.0f} Hz,差 {abs(m-self.fps)/self.fps*100:.0f}%")
        if self.dropped_frames:
            n.append(f"相机少给了 {self.dropped_frames} 帧(那些时刻整帧丢弃,不影响已存的)")
        return n

    # ── 落盘 ──────────────────────────────────────────────────────────────
    def save(self, index, task=None, success=True):
        """把缓冲交给后台线程,立刻返回。缓冲当场换新的,可以接着录。"""
        self.flush()
        st = np.asarray(self.stamps, dtype=np.float64)
        payload = dict(
            state=np.stack(self.state),
            action=np.stack(self.action),
            ee_pose=np.stack(self.ee_pose),
            ee_target=np.stack(self.ee_target),
            task=task if task is not None else self.task,
            fps=self.fps,
            timestamp=st - st[0] if len(st) else st,
            fps_measured=float(self.measured_fps() or self.fps),
            camera_keys=np.array(self.camera_keys),
            arm_dof=ARM_DOF, hand_dof=HAND_DOF,
            # 这条算不算做成了。训练时按它筛，不用靠人记。
            success=bool(success),
        )
        frames, self.frames = self.frames, {k: [] for k in self.camera_keys}
        path = self.out_dir / f"episode_{index:04d}.npz"
        self.reset()
        self._writer = threading.Thread(target=self._write, args=(path, payload, frames),
                                        name=f"rec-{index:04d}", daemon=False)
        self._writer.start()
        return path

    def _write(self, path, payload, frames):
        try:
            import cv2
            for key, imgs in frames.items():
                if not imgs:
                    continue
                h, w = imgs[0].shape[:2]
                vw = cv2.VideoWriter(str(path.with_name(f"{path.stem}_{key}.mp4")),
                                     cv2.VideoWriter_fourcc(*self.codec),
                                     payload["fps_measured"], (w, h))
                for f in imgs:
                    vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))   # 驱动给 RGB,写视频要 BGR
                vw.release()
            np.savez_compressed(path, **payload)
        except Exception as exc:                # noqa: BLE001
            self._write_error = f"{path.name} 写盘失败:{exc}"

    def flush(self, timeout=90.0):
        if self._writer is not None:
            self._writer.join(timeout)
            if self._writer.is_alive():
                print(f"  ⚠ 上一条还在写盘(超过 {timeout:.0f}s)")
            self._writer = None
        if self._write_error:
            print(f"  ✗ {self._write_error}")
            self._write_error = None


def next_index(out_dir):
    """已有几条就从几号接着编,不覆盖。"""
    out_dir = Path(out_dir).expanduser()
    if not out_dir.is_dir():
        return 0
    used = [int(p.stem.split("_")[1]) for p in out_dir.glob("episode_*.npz")
            if p.stem.split("_")[1].isdigit()]
    return max(used) + 1 if used else 0
