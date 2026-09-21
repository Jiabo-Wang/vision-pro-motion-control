#!/usr/bin/env python3
"""认相机:列出所有 by-path 设备,各抓一张图存下来,你看图就知道哪路是哪路。

两台相机型号和序列号都一样(都报 SN0001),`/dev/v4l/by-id/` 只有一条,
`video0` / `video2` 拔插后会互换。**只能用 `/dev/v4l/by-path/`**,它绑在物理插口上。
每台相机还占一个只有元数据的第二节点(video1/video3),用 OpenCV 打开会卡死,
只有 `-index0` 是彩色流。

    python probe_cameras.py              # 列设备 + 各抓一张存到 /tmp
    python probe_cameras.py --view       # 再开窗口实时看(需要显示)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def list_paths():
    d = Path("/dev/v4l/by-path")
    if not d.is_dir():
        return []
    # 只要 -index0(彩色流),usbv2 是同一个设备的别名,去重
    seen, out = set(), []
    for p in sorted(d.iterdir()):
        if not p.name.endswith("-video-index0"):
            continue
        real = p.resolve()
        if real in seen:
            continue
        seen.add(real)
        out.append((str(p), str(real)))
    return out


def main():
    ap = argparse.ArgumentParser(description="认相机:哪个 by-path 是顶部,哪个是腕部")
    ap.add_argument("--view", action="store_true", help="抓完再开窗口实时看")
    ap.add_argument("--out", default="/tmp", help="截图存哪")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()

    paths = list_paths()
    if not paths:
        print("没找到 /dev/v4l/by-path 下的彩色流设备。相机插上了吗?")
        return 1

    print(f"找到 {len(paths)} 路彩色流:\n")
    caps = []
    for i, (path, real) in enumerate(paths):
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        ok, frame = (False, None)
        for _ in range(10):                 # 头几帧常是空的
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        shot = Path(a.out) / f"cam{i}.jpg"
        if ok and frame is not None:
            cv2.imwrite(str(shot), frame)
            h, w = frame.shape[:2]
            print(f"  [{i}] {path}")
            print(f"      -> {real}   {w}x{h}   截图 {shot}")
        else:
            print(f"  [{i}] {path}\n      -> 打不开或没出图")
        if a.view and ok:
            caps.append((f"cam{i}", cap))
        else:
            cap.release()

    print("\n看一眼截图,确认哪张是顶部俯视、哪张是腕部视角,然后这样跑遥操:")
    if len(paths) >= 2:
        print(f"\n  python avp_servoj_teleop.py <头显IP> --planner spline --lag-ms 50 \\")
        print(f"      --record --task \"pick up the bolt\" \\")
        print(f"      --cam-top   {paths[0][0]} \\")
        print(f"      --cam-wrist {paths[1][0]}")
        print("\n(如果截图显示反了,把上面两行的路径对调)")

    if a.view and caps:
        print("\n实时预览,按 q 关闭")
        while True:
            for name, cap in caps:
                ok, f = cap.read()
                if ok:
                    cv2.imshow(name, f)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        for _, cap in caps:
            cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
