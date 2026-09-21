#!/usr/bin/env python3
"""回放采好的数据，用眼睛确认这条能不能用。只读文件，不碰硬件。

    python replay_episode.py                       # 列出全部，一行一条
    python replay_episode.py --episode 7           # 只看第 7 条
    python replay_episode.py --all                 # 从头一条条看
    python replay_episode.py --all --only-flagged  # 只看自动检查报了问题的
    python replay_episode.py --episode 7 --cam wrist   # 只看腕部那一路

两路相机默认并排显示。窗口里：

    空格   暂停 / 继续
    ← →    上一帧 / 下一帧（暂停时用）
    a d    上一条 / 下一条
    q      退出

画面底部六条绿色是六根手指的开合，0 是握紧、1000 是张开。
顶部一行是这一帧末端走了多少毫米。

**自动检查查不出来的两件事，只能在这里看**：抓稳没有、放到位没有。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HAND_NAMES = ["little", "ring", "middle", "index", "th_bend", "th_rot"]
ARM_DOF = 7


# ── 读 ────────────────────────────────────────────────────────────────────
def load_episode(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    cams = [str(k) for k in d["camera_keys"]]
    # 回放要用**实测**帧率。npz 里的 `fps` 是命令行给的标称值，回路没跑到那么快时
    # 它是错的，按它放会比真实快。
    nominal = float(d["fps"])
    measured = float(d["fps_measured"]) if "fps_measured" in d else None
    return {
        "path": npz_path,
        "state": d["state"], "action": d["action"],
        "ee_pose": d["ee_pose"],
        "task": str(d["task"]),
        "success": bool(d["success"]) if "success" in d else None,
        "fps": measured or nominal, "fps_nominal": nominal, "fps_measured": measured,
        "camera_keys": cams,
        "videos": {k: (npz_path.with_name(f"{npz_path.stem}_{k}.mp4")
                       if npz_path.with_name(f"{npz_path.stem}_{k}.mp4").exists() else None)
                   for k in cams},
    }


def summarise(ep: dict) -> str:
    pose, act = ep["ee_pose"], ep["action"]
    # 走过的**路径长度**，不是首尾直线距离。抓放任务做完臂就回到起点附近，
    # 用直线距离会把好数据误判成没动。
    seg = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1) * 1000
    path_mm, hand = float(seg.sum()), act[:, ARM_DOF:]
    span = float(np.max(hand.max(0) - hand.min(0)))
    notes = []
    if span < 20:
        notes.append("手没动")
    if path_mm < 100:
        notes.append("臂基本没动")
    if len(act) < 60:
        notes.append("太短")
    if ep["success"] is False:
        notes.append("没标成功")
    flag = ("  ⚠ " + "、".join(notes)) if notes else ""
    return (f"{ep['path'].name:22s} {len(act):4d} 帧 {len(act)/ep['fps']:5.1f}s  "
            f"末端走 {path_mm:5.0f}mm  逐帧中位 {np.median(seg) if len(seg) else 0:4.2f}mm  "
            f"手行程 {span:4.0f}{flag}")


def read_frames(video_path):
    import cv2
    if video_path is None:
        return []
    cap, frames = cv2.VideoCapture(str(video_path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)                    # 保持 BGR，直接给 imshow
    cap.release()
    return frames


# ── 画 ────────────────────────────────────────────────────────────────────
def panel(frame, name, height=480):
    import cv2
    h, w = frame.shape[:2]
    if h != height:
        frame = cv2.resize(frame, (int(w * height / h), height))
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, name, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    return out


def compose(frames_by_cam, order, i):
    import cv2
    panels = []
    for name in order:
        fr = frames_by_cam.get(name) or []
        if i < len(fr):
            panels.append(panel(fr[i], name))
        else:
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, f"{name}: 没有这一帧", (60, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 200), 2)
            panels.append(panel(blank, name))
    return np.hstack(panels) if panels else np.zeros((480, 640, 3), np.uint8)


def annotate(frame, ep, i):
    import cv2
    out = frame.copy()
    h, w = out.shape[:2]
    pose = ep["ee_pose"]
    step_mm = (float(np.linalg.norm(pose[i, :3] - pose[i - 1, :3])) * 1000) if i else 0.0
    hand = ep["action"][i, ARM_DOF:]

    cv2.rectangle(out, (0, h - 64), (w, h), (0, 0, 0), -1)
    tag = {True: "成功", False: "未标成功", None: ""}[ep["success"]]
    cv2.putText(out, f"{ep['path'].stem}  {i+1}/{len(ep['action'])}  "
                     f"步长 {step_mm:5.2f} mm  {tag}",
                (8, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 六根手指各一条：0 握紧、1000 张开
    bw = (w - 24) // 6
    for k in range(6):
        x0 = 12 + k * bw
        cv2.rectangle(out, (x0, h - 34), (x0 + bw - 6, h - 18), (70, 70, 70), 1)
        fill = int((bw - 8) * float(np.clip(hand[k] / 1000.0, 0, 1)))
        if fill > 0:
            cv2.rectangle(out, (x0 + 1, h - 33), (x0 + 1 + fill, h - 19), (0, 200, 0), -1)
        cv2.putText(out, HAND_NAMES[k], (x0 + 2, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (190, 190, 190), 1)

    cv2.line(out, (0, h - 66), (int(w * (i + 1) / len(ep["action"])), h - 66), (0, 165, 255), 2)
    return out


def play(ep: dict, only=None) -> str:
    """回放一条。返回 'next' / 'prev' / 'quit'。"""
    import cv2

    print(f"\n  {ep['path'].name}   「{ep['task']}」")
    print(f"  {summarise(ep)}")
    print(f"  帧率 实测 {ep['fps_measured']:.1f} Hz（标称 {ep['fps_nominal']:.0f}）"
          if ep["fps_measured"] else f"  帧率 {ep['fps_nominal']:.0f} Hz（标称，这条没记时间戳）")

    order = [k for k in ep["camera_keys"] if only is None or k == only]
    if only is not None and not order:
        print(f"  这条没有 '{only}'，有的是 {ep['camera_keys']}")
        return "next"

    frames = {k: read_frames(ep["videos"].get(k)) for k in order}
    have = {k: len(v) for k, v in frames.items()}
    print(f"  相机 {have}")
    if [k for k, v in have.items() if v == 0]:
        print(f"  ⚠ 这几路没有视频：{[k for k, v in have.items() if v == 0]}")

    n, longest = len(ep["action"]), max(have.values(), default=0)
    if longest == 0:
        print("  没有视频，只能看数字：")
        for i in range(0, n, max(1, n // 20)):
            p = ep["ee_pose"][i, :3] * 1000
            print(f"    {i:4d}  末端 ({p[0]:+6.0f},{p[1]:+6.0f},{p[2]:+6.0f})mm  "
                  f"手 {np.round(ep['action'][i, ARM_DOF:]).astype(int).tolist()}")
        return "next"
    if longest < n:
        print(f"  视频只有 {longest} 帧，数据有 {n} 帧，按短的放")
        n = longest

    win = "replay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, 640 * len(order)), 480)
    delay = max(1, int(1000 / ep["fps"]))
    i, paused = 0, False
    while True:
        cv2.imshow(win, annotate(compose(frames, order, i), ep, i))
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(win); return "quit"
        if key == ord("d"):
            cv2.destroyWindow(win); return "next"
        if key == ord("a"):
            cv2.destroyWindow(win); return "prev"
        if key == ord(" "):
            paused = not paused
        if key in (81, ord(",")):                 # ←
            i, paused = max(0, i - 1), True
        if key in (83, ord(".")):                 # →
            i, paused = min(n - 1, i + 1), True
        if not paused:
            i += 1
            if i >= n:
                cv2.destroyWindow(win); return "next"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="~/atom_episodes", help="数据在哪")
    ap.add_argument("--episode", type=int, help="看第几条（就是文件名里的编号）")
    ap.add_argument("--all", action="store_true", help="从头一条条看")
    ap.add_argument("--only-flagged", action="store_true", help="只看列表里带 ⚠ 的")
    ap.add_argument("--cam", help="只看某一路（top / wrist）。默认全部并排")
    a = ap.parse_args()

    src = Path(a.src).expanduser()
    paths = sorted(src.glob("episode_*.npz"))
    if not paths:
        print(f"{src} 里没有数据")
        return 1

    if a.episode is None and not a.all:
        print(f"{src}   共 {len(paths)} 条\n")
        bad = 0
        for p in paths:
            line = summarise(load_episode(p))
            print("  " + line)
            bad += "⚠" in line
        print(f"\n  {len(paths)} 条里 {bad} 条带标记")
        print(f"  看某一条：--episode 7    挨个看：--all    只看带标记的：--all --only-flagged")
        return 0

    if a.only_flagged:
        paths = [p for p in paths if "⚠" in summarise(load_episode(p))]
        if not paths:
            print("  没有带标记的，全都正常")
            return 0
        print(f"  {len(paths)} 条带标记")

    # --episode 给的是文件名里的编号，不是列表下标。删过数据后两者会对不上。
    if a.episode is not None:
        want = src / f"episode_{a.episode:04d}.npz"
        if want not in paths:
            print(f"  没有 {want.name}。现有的是 "
                  f"{[p.stem.split('_')[1] for p in paths][:20]}{' ...' if len(paths) > 20 else ''}")
            return 1
        index = paths.index(want)
    else:
        index = 0

    while 0 <= index < len(paths):
        act = play(load_episode(paths[index]), only=a.cam)
        if act == "quit":
            break
        if act == "prev":
            index -= 1
        else:
            if not (a.all or a.only_flagged):
                break
            index += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
