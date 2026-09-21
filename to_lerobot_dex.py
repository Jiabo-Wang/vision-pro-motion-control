#!/usr/bin/env python3
"""把采到的 episode 转成 lerobot 数据集。13 自由度版本(AR5 七轴 + 因时手六维)。

**要在装了 lerobot 的环境里跑**,不是遥操那个环境。遥操环境的 torch 版本是配死的,
装 lerobot 会动到它。

    conda activate lerobot
    python to_lerobot_dex.py --src ~/atom_episodes --repo-id atom_ar5_bolt

时间戳用 npz 里的 `fps_measured`,不是标称的 30 —— 回路实际跑多快事后只能靠它。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ARM_NAMES = [f"joint_{i}" for i in range(7)]
HAND_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
STATE_NAMES = ARM_NAMES + HAND_NAMES              # 13
ACTION_NAMES = [f"cmd_{n}" for n in STATE_NAMES]  # 13
EE_NAMES = ["x", "y", "z", "rx", "ry", "rz"]      # 位置 m + 轴角 rad


def read_video(path, n_expect):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != n_expect:
        print(f"    ⚠ {path.name}: {len(frames)} 帧，npz 说 {n_expect} 帧")
    return frames


def build_features(camera_keys, h, w):
    f = {
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
        "observation.ee_pose": {"dtype": "float32", "shape": (len(EE_NAMES),), "names": EE_NAMES},
    }
    for k in camera_keys:
        f[f"observation.images.{k}"] = {"dtype": "video", "shape": (h, w, 3),
                                        "names": ["height", "width", "channels"]}
    return f


def main():
    ap = argparse.ArgumentParser(description="采集目录 → lerobot 数据集（13 自由度）")
    ap.add_argument("--src", required=True, help="录制输出目录")
    ap.add_argument("--repo-id", required=True, help="数据集名字")
    ap.add_argument("--root", default="", help="数据集存哪，留空用 lerobot 默认")
    ap.add_argument("--robot-type", default="ar5_inspire")
    a = ap.parse_args()

    src = Path(a.src).expanduser()
    eps = sorted(src.glob("episode_*.npz"))
    if not eps:
        print(f"{src} 下没有 episode_*.npz")
        return 1

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    d0 = np.load(eps[0], allow_pickle=True)
    cams = [str(x) for x in d0["camera_keys"]]
    n0 = len(d0["state"])
    probe = read_video(eps[0].with_name(f"{eps[0].stem}_{cams[0]}.mp4"), n0)
    if not probe:
        print("第一条的视频读不出来，检查 mp4")
        return 1
    h, w = probe[0].shape[:2]
    fps = float(d0["fps_measured"])
    print(f"{len(eps)} 条  相机 {cams}  {w}x{h}  参考帧率 {fps:.1f}Hz")

    ds = LeRobotDataset.create(
        repo_id=a.repo_id, fps=int(round(fps)), robot_type=a.robot_type,
        features=build_features(cams, h, w),
        root=a.root or None, use_videos=True,
    )

    for p in eps:
        d = np.load(p, allow_pickle=True)
        state, action, ee = d["state"], d["action"], d["ee_pose"]
        n = len(state)
        vids = {k: read_video(p.with_name(f"{p.stem}_{k}.mp4"), n) for k in cams}
        m = min([n] + [len(v) for v in vids.values()])
        if m < n:
            print(f"  {p.name}: 按最短的 {m} 帧对齐")
        for i in range(m):
            frame = {
                "observation.state": state[i].astype(np.float32),
                "action": action[i].astype(np.float32),
                "observation.ee_pose": ee[i].astype(np.float32),
            }
            for k in cams:
                frame[f"observation.images.{k}"] = vids[k][i]
            ds.add_frame(frame, task=str(d["task"]))
        ds.save_episode()
        print(f"  {p.name} -> {m} 帧")

    print(f"\n完成，{len(eps)} 条。数据集 {a.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
