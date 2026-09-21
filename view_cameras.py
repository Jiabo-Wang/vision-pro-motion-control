#!/usr/bin/env python3
"""开录之前看一眼两路相机：画面对不对、摆得正不正、会不会掉帧。

    python view_cameras.py                 # 两路并排实时看
    python view_cameras.py --check         # 不开窗口，抓 60 帧出一份体检（ssh 进来时用）
    python view_cameras.py --list          # 只列设备路径，拿去填 --cam-top / --cam-wrist

窗口里：
    q / Esc   退出
    g         叠三分线和中心十字，摆正相机用
    s         两路各存一张 PNG 到当前目录
    1 2       只看第一路 / 第二路，再按一次回到并排

**设备一律用 `/dev/v4l/by-path/...`，不要用 video0 / video2。** 两台相机型号和
序列号一样（都报 SN0001），`by-id` 只剩一条，`videoN` 拔插一次就会互换，
结果是腕部画面被喂成俯视画面，数据全废。by-path 绑的是物理插口，不会变。

每台相机还占一个只有元数据的第二节点（video1 / video3），用 OpenCV 打开会卡死。
只有 `-index0` 是彩色流，下面的扫描已经筛过了。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def list_paths():
    """/dev/v4l/by-path 下所有彩色流，去掉 usbv2 别名。"""
    d = Path("/dev/v4l/by-path")
    if not d.is_dir():
        return []
    seen, out = set(), []
    for p in sorted(d.iterdir()):
        if not p.name.endswith("-video-index0"):
            continue
        real = p.resolve()
        if real in seen:
            continue
        seen.add(real)
        out.append(str(p))
    return out


def disable_dynamic_framerate(path):
    """让相机别拿帧率换曝光。

    这个开关开着的时候，室内正常光照下相机只给 19 Hz，换曝光时间、换像素格式
    都救不回来；关掉就能到 30。它在 UVC 驱动里默认是关的，但这台机器上开机是开的。
    OpenCV 没有对应的 property，只能走 v4l2-ctl。
    """
    import subprocess
    try:
        subprocess.run(["v4l2-ctl", "-d", path, "--set-ctrl",
                        "exposure_dynamic_framerate=0"], capture_output=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        pass


def open_cam(path, width, height, fps):
    import cv2
    disable_dynamic_framerate(path)
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # **两个缓冲，不是一个。** 只给一个缓冲时，取图和还缓冲之间驱动没地方放下一帧，
    # 于是每隔一帧丢一帧 —— 实测 15.5 Hz 对 29.3 Hz，两路同时也一样。
    # 代价是最多晚一帧，可以接受。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    for _ in range(10):                     # 头几帧常是空的
        ok, f = cap.read()
        if ok and f is not None:
            return cap, f
    cap.release()
    return None, None


def draw_grid(frame):
    import cv2
    h, w = frame.shape[:2]
    for i in (1, 2):
        cv2.line(frame, (w * i // 3, 0), (w * i // 3, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h * i // 3), (w, h * i // 3), (0, 255, 0), 1)
    cv2.drawMarker(frame, (w // 2, h // 2), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)


def bar(frame, text):
    import cv2
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def health(caps, paths, n):
    """抓 n 帧，报分辨率、实测帧率、黑帧、冻帧。冻帧 = 相邻两帧逐像素完全相同。"""
    import cv2
    print(f"\n  抓 {n} 帧体检 ...")
    bad = 0
    for k, (cap, path) in enumerate(zip(caps, paths)):
        prev, froze, black, t0, got = None, 0, 0, time.time(), 0
        for _ in range(n):
            ok, f = cap.read()
            if not ok or f is None:
                continue
            got += 1
            g = cv2.cvtColor(cv2.resize(f, (160, 120)), cv2.COLOR_BGR2GRAY)
            if prev is not None and np.array_equal(g, prev):
                froze += 1
            if g.mean() < 10 or g.std() < 3:
                black += 1
            prev = g
        hz = got / max(time.time() - t0, 1e-6)
        h, w = f.shape[:2] if f is not None else (0, 0)
        flags = []
        if got < n:
            flags.append(f"只读到 {got}/{n} 帧")
        if froze:
            flags.append(f"{froze} 帧和上一帧完全相同（画面冻住）")
        if black:
            flags.append(f"{black} 黑帧（镜头盖着？曝光没上来？）")
        if hz < 20:
            flags.append(f"只有 {hz:.0f} Hz，低于录制的 30 Hz")
        bad += bool(flags)
        print(f"  cam{k}  {w}x{h}  {hz:5.1f} Hz   {path}")
        for t in flags:
            print(f"        ⚠ {t}")
        if not flags:
            print(f"        ✓ 正常")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="只列设备路径")
    ap.add_argument("--check", action="store_true", help="不开窗口，抓几帧出体检报告")
    ap.add_argument("--frames", type=int, default=60, help="--check 抓多少帧")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--save", default=".", help="按 s 存图存到哪")
    a = ap.parse_args()

    import cv2

    paths = list_paths()
    if not paths:
        print("  /dev/v4l/by-path 下没有彩色流。相机插上了吗？`lsusb` 看得到吗？")
        return 1

    print(f"  找到 {len(paths)} 路彩色流：")
    for k, p in enumerate(paths):
        print(f"    cam{k}  {p}")
    if a.list:
        if len(paths) >= 2:
            print(f"\n  填进遥操命令：\n      --cam-top   {paths[0]} \\\n      --cam-wrist {paths[1]}")
            print(f"  **先看过画面再决定哪个是 top、哪个是 wrist**，两台相机型号一样，"
                  f"从路径看不出来。")
        return 0

    caps, ok_paths = [], []
    for p in paths:
        cap, _ = open_cam(p, a.width, a.height, a.fps)
        if cap is None:
            print(f"  ✗ 打不开或没出图：{p}")
            print(f"    被别的进程占着？ fuser -v {p}")
            continue
        caps.append(cap)
        ok_paths.append(p)
    if not caps:
        print("  一路都没打开")
        return 1

    try:
        if a.check:
            bad = health(caps, ok_paths, a.frames)
            print(f"\n  {'全部正常' if not bad else f'{bad} 路有问题'}")
            return 0 if not bad else 2

        print(f"\n  q 退出   g 网格   s 存图   1/2 单看一路")
        print(f"  **确认两件事**：哪一路是俯视、哪一路是腕部；桌面和物体都在画面里。")
        grid, solo, shown, t0, hz = False, None, 0, time.time(), 0.0
        win = "cameras"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(1600, a.width * len(caps)), a.height)
        while True:
            panels = []
            for k, cap in enumerate(caps):
                ok, f = cap.read()
                if not ok or f is None:
                    f = np.zeros((a.height, a.width, 3), np.uint8)
                    cv2.putText(f, f"cam{k} 读帧失败", (40, a.height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 220), 2)
                if solo is not None and k != solo:
                    continue
                v = f.copy()
                if grid:
                    draw_grid(v)
                bar(v, f"cam{k}   {f.shape[1]}x{f.shape[0]}   {hz:.1f} fps")
                panels.append(v)
            cv2.imshow(win, np.hstack(panels) if panels else np.zeros((10, 10, 3), np.uint8))

            shown += 1
            if shown % 15 == 0:
                now = time.time()
                hz, t0 = 15.0 / (now - t0), now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                grid = not grid
            if key in (ord("1"), ord("2")):
                want = key - ord("1")
                solo = None if solo == want else (want if want < len(caps) else None)
            if key == ord("s"):
                stamp = time.strftime("%H%M%S")
                for k, cap in enumerate(caps):
                    ok, f = cap.read()
                    if ok:
                        out = Path(a.save).expanduser() / f"cam{k}_{stamp}.png"
                        cv2.imwrite(str(out), f)
                        print(f"  已存 {out}")
    finally:
        for c in caps:
            c.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
