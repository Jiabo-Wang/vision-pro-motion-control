#!/usr/bin/env python3
"""删条目、补编号、查孤儿文件。只动文件，不碰硬件。

一条 episode 是**一个 npz 加每路相机一个 mp4**，手动 rm 很容易只删掉 npz，
留下的 mp4 会让后面的检查和转换对不上。用这个删。

    python manage_episodes.py                      # 列出现状：有哪些编号、缺哪些、有没有孤儿
    python manage_episodes.py --delete 3,7,12      # 先看要删什么（默认只看不动）
    python manage_episodes.py --delete 3,7,12 --apply
    python manage_episodes.py --renumber --apply   # 把空洞补上，变成连续编号

**编号该不该重排：采集过程中不要排。** 理由是三条：

  * 有空洞不影响任何下游。`to_lerobot_dex.py` 按文件名排序读，lerobot 自己重新
    编 episode_index，空洞会被自动吃掉。
  * 重排会让你手上的记录失效。回放时记下「第 12 条抓歪了」，一重排 12 号就是
    另一条了。
  * 重排是批量改名，中途断电就是一地半新半旧的文件名。

要排就等这一批彻底采完、检查完、转完数据集之后再排一次，而且带 --apply 前
先看一遍 dry-run 的映射表。

**存储文件名不用改。** `episode_%04d.npz` 加 `episode_%04d_<相机>.mp4` 这个格式
已经够用：编号定位、相机名在后缀、npz 里另存了 task / success / 时间戳。把任务名
或日期塞进文件名反而会让删改和重排都变麻烦，那些信息在 npz 里查得到。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def episodes(src: Path):
    """返回 [(编号, npz 路径, [mp4 路径...])]，按编号排序。"""
    out = []
    for p in sorted(src.glob("episode_*.npz")):
        tag = p.stem.split("_")[1]
        if not tag.isdigit():
            continue
        out.append((int(tag), p, sorted(src.glob(f"{p.stem}_*.mp4"))))
    return out


def orphans(src: Path, eps):
    """有 mp4 没有对应 npz —— 手动 rm 最容易留下的就是这个。"""
    known = {m for _, _, mp4s in eps for m in mp4s}
    return [m for m in sorted(src.glob("episode_*.mp4")) if m not in known]


def show(src: Path, eps):
    print(f"{src}   {len(eps)} 条")
    if not eps:
        return
    nums = [n for n, _, _ in eps]
    holes = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
    for n, npz, mp4s in eps:
        try:
            d = np.load(npz, allow_pickle=True)
            nf = len(d["state"])
            sc = {True: "成功", False: "未标成功"}.get(
                bool(d["success"]) if "success" in d else None, "—")
            task = str(d["task"])[:24]
        except Exception as e:                      # noqa: BLE001
            nf, sc, task = -1, "读不出", f"{type(e).__name__}"
        print(f"  {n:4d}  {nf:4d} 帧  {len(mp4s)} 路视频  {sc:8s} 「{task}」")
    print(f"\n  编号 {nums[0]}~{nums[-1]}"
          + (f"   空洞 {holes}" if holes else "   连续，没有空洞"))
    orph = orphans(src, eps)
    if orph:
        print(f"  ⚠ {len(orph)} 个孤儿 mp4（没有对应的 npz）：{[m.name for m in orph[:8]]}")
        print(f"    这些是手动 rm 只删了 npz 留下的，会让检查和转换对不上。删掉：")
        print(f"      python manage_episodes.py --src {src} --clean-orphans --apply")


def do_delete(src: Path, eps, want, apply):
    idx = {n: (npz, mp4s) for n, npz, mp4s in eps}
    missing = [n for n in want if n not in idx]
    if missing:
        print(f"  这些编号不存在，忽略：{missing}")
    todo = [n for n in want if n in idx]
    if not todo:
        return 1
    files = []
    for n in todo:
        npz, mp4s = idx[n]
        files += [npz] + mp4s
    print(f"  要删 {len(todo)} 条，共 {len(files)} 个文件：")
    for f in files:
        print(f"    {f.name}")
    if not apply:
        print(f"\n  这是预览。真删加 --apply")
        return 0
    for f in files:
        f.unlink()
    print(f"\n  删了 {len(files)} 个文件。编号会留下空洞，**不用管**，下游不受影响。")
    return 0


def do_renumber(src: Path, eps, start, apply):
    plan = []
    for k, (n, npz, mp4s) in enumerate(eps):
        new = start + k
        if new == n:
            continue
        plan.append((n, new, [npz] + mp4s))
    if not plan:
        print(f"  已经是从 {start} 开始的连续编号，不用动")
        return 0
    print(f"  {len(plan)} 条要改名：")
    for old, new, files in plan:
        print(f"    {old:4d} -> {new:4d}   {len(files)} 个文件")
    if not apply:
        print(f"\n  这是预览。真改加 --apply。改之前先确认这一批已经检查完、转过数据集了。")
        return 0

    # 两步改名。直接 0->0, 1->0 这种会撞车，先全改成临时名再落位。
    tmp = []
    for old, new, files in plan:
        for f in files:
            t = f.with_name(f".renum_{f.name}")
            f.rename(t)
            tmp.append((t, f.name.replace(f"episode_{old:04d}", f"episode_{new:04d}")))
    for t, final in tmp:
        t.rename(t.with_name(final))
    log = src / "renumber_log.txt"
    with log.open("a") as fh:
        for old, new, _ in plan:
            fh.write(f"{old:04d} -> {new:04d}\n")
    print(f"\n  改完 {len(tmp)} 个文件。映射记在 {log.name}，回放笔记按它对一遍。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="~/atom_episodes")
    ap.add_argument("--delete", default="", help="要删的编号，逗号隔开，如 3,7,12")
    ap.add_argument("--renumber", action="store_true", help="把空洞补上，改成连续编号")
    ap.add_argument("--start", type=int, default=0, help="重排从几号开始")
    ap.add_argument("--clean-orphans", action="store_true", help="删掉没有 npz 的孤儿 mp4")
    ap.add_argument("--apply", action="store_true", help="真的动文件。不加就只预览")
    a = ap.parse_args()

    src = Path(a.src).expanduser()
    if not src.is_dir():
        print(f"{src} 不存在")
        return 1
    eps = episodes(src)

    if a.delete:
        try:
            want = sorted({int(x) for x in a.delete.replace(" ", "").split(",") if x})
        except ValueError:
            print(f"  --delete 要的是数字，逗号隔开，比如 3,7,12")
            return 1
        return do_delete(src, eps, want, a.apply)

    if a.clean_orphans:
        orph = orphans(src, eps)
        if not orph:
            print("  没有孤儿文件")
            return 0
        print(f"  {len(orph)} 个孤儿 mp4：")
        for m in orph:
            print(f"    {m.name}")
        if not a.apply:
            print(f"\n  这是预览。真删加 --apply")
            return 0
        for m in orph:
            m.unlink()
        print(f"\n  删了 {len(orph)} 个")
        return 0

    if a.renumber:
        if not eps:
            print(f"  {src} 里没有数据")
            return 1
        return do_renumber(src, eps, a.start, a.apply)

    show(src, eps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
