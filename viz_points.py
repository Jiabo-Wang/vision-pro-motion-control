#!/usr/bin/env python3
"""把 avp_movel_teleop.py 导出的逐点 CSV 画成流水线动画 + 汇总表。

用法：
    python viz_points.py logs/points_20260921_104500.csv          # 动画 + 表
    python viz_points.py logs/points_*.csv --table-only           # 只打表
    python viz_points.py logs/points_x.csv --out /tmp/pipe.mp4    # 指定输出

动画上半部分是流水线：每个点是一个方块，从「头显帧」依次流到「滤波」「本地队列」
「控制器队列」「已走完」，方块停在哪一栏就说明当时它卡在哪一环。下半部分是同一
时刻的计数曲线和 TCP 轨迹。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

STAGES = ("frame", "filt", "queue", "push", "done")
STAGE_CN = {"frame": "头显帧", "filt": "滤波", "queue": "本地队列", "push": "控制器队列", "done": "已走完"}


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                if v == "" or v is None:
                    d[k] = None
                elif k in ("pid", "cid", "batch_size", "inflight_at_push"):
                    d[k] = int(float(v))
                elif k == "fate":
                    d[k] = v
                else:
                    d[k] = float(v)
            rows.append(d)
    return rows


def stage_of(row, t):
    """t 时刻这个点处在哪一环节；还没产生返回 None。"""
    if row["t_frame"] is None or t < row["t_frame"]:
        return None
    if row["t_done"] is not None and t >= row["t_done"]:
        return "done"
    if row["t_push"] is not None and t >= row["t_push"]:
        return "push"
    if row["t_queue"] is not None and t >= row["t_queue"]:
        return "queue"
    if row["t_filt"] is not None and t >= row["t_filt"]:
        return "filt"
    return "frame"


def table(rows, path):
    done = [r for r in rows if r["t_done"] is not None]
    def med(a, b, src=done):
        v = [r[b] - r[a] for r in src if r[a] is not None and r[b] is not None]
        return np.median(v) * 1000 if v else float("nan")
    def p95(a, b, src=done):
        v = [r[b] - r[a] for r in src if r[a] is not None and r[b] is not None]
        return np.percentile(v, 95) * 1000 if v else float("nan")
    fates = {}
    for r in rows:
        fates[r["fate"]] = fates.get(r["fate"], 0) + 1
    span = max((r["t_frame"] for r in rows if r["t_frame"] is not None), default=0)
    print(f"\n=== {Path(path).name} ===")
    print(f"时长 {span:.1f}s   采样点 {len(rows)}   去向: " +
          "  ".join(f"{k}={v}" for k, v in sorted(fates.items())))
    print(f"\n{'环节':<22}{'中位 ms':>10}{'p95 ms':>10}")
    print("-" * 42)
    for a, b, name in (("t_frame", "t_filt", "头显帧 → 滤波完成"),
                       ("t_filt", "t_queue", "滤波 → 进本地队列"),
                       ("t_queue", "t_push", "本地队列 → 下发"),
                       ("t_push", "t_done", "下发 → 控制器走完"),
                       ("t_frame", "t_done", "合计（端到端）")):
        print(f"{name:<20}{med(a,b):>10.0f}{p95(a,b):>10.0f}")
    if done:
        segs = [r["seg_mm"] for r in done if r["seg_mm"] is not None]
        sps = [r["speed"] for r in done if r["speed"] is not None]
        bs = [r["batch_size"] for r in done if r["batch_size"] is not None]
        infl = [r["inflight_at_push"] for r in done if r["inflight_at_push"] is not None]
        print(f"\n段长 中位 {np.median(segs):.0f}mm   段速 中位 {np.median(sps):.0f}mm/s   "
              f"每批点数 中位 {np.median(bs):.0f}   下发时在飞 中位 {np.median(infl):.0f}")
        exec_t = [(r["t_done"] - r["t_push"]) for r in done]
        theory = [s / max(v, 1) for s, v in zip(segs, sps)]
        print(f"实测执行 中位 {np.median(exec_t)*1000:.0f}ms   按段长/段速的理论值 {np.median(theory)*1000:.0f}ms   "
              f"额外开销 {np.median(exec_t)*1000 - np.median(theory)*1000:.0f}ms")


def animate(rows, out, fps=20, speed=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    for fam in ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"):
        try:
            matplotlib.rcParams["font.sans-serif"] = [fam] + matplotlib.rcParams["font.sans-serif"]
            break
        except Exception:  # noqa: BLE001
            pass
    matplotlib.rcParams["axes.unicode_minus"] = False
    if not matplotlib.rcParams.get("animation.ffmpeg_path", "ffmpeg").startswith("/"):
        try:                                   # 本机没有系统 ffmpeg，用 pip 装的那个
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            pass

    t_end = max(r[k] for r in rows for k in ("t_frame", "t_done") if r[k] is not None)
    times = np.arange(0, t_end + 0.5, speed / fps)
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.28, wspace=0.22)
    axp = fig.add_subplot(gs[0, :])
    axc = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[1, 1])

    axp.set_xlim(-0.5, len(STAGES) - 0.5)
    axp.set_xticks(range(len(STAGES)))
    axp.set_xticklabels([STAGE_CN[s] for s in STAGES])
    axp.set_ylim(0, 12)
    axp.set_yticks([])
    axp.set_title("点的流水线（每个方块 = 一个采样点）")
    for i in range(len(STAGES)):
        axp.axvspan(i - 0.42, i + 0.42, color="0.94", zorder=0)
    scat = axp.scatter([], [], s=90, marker="s")
    counts_txt = [axp.text(i, 11.3, "", ha="center", fontsize=11, weight="bold") for i in range(len(STAGES))]
    clock = axp.text(0.01, 0.96, "", transform=axp.transAxes, va="top", fontsize=11)

    # 计数曲线
    cnt = {s: [] for s in STAGES}
    for t in times:
        c = {s: 0 for s in STAGES}
        for r in rows:
            s = stage_of(r, t)
            if s:
                c[s] += 1
        for s in STAGES:
            cnt[s].append(c[s])
    axc.set_title("各环节累计/驻留点数")
    axc.set_xlabel("时间 s")
    lines = {}
    for s in STAGES:
        (ln,) = axc.plot([], [], label=STAGE_CN[s])
        lines[s] = ln
    axc.set_xlim(0, t_end)
    axc.set_ylim(0, max(max(v) for v in cnt.values()) * 1.1 + 1)
    axc.legend(fontsize=8, ncol=2)

    xs = [r["x"] for r in rows if r["x"] is not None]
    ys = [r["y"] for r in rows if r["y"] is not None]
    axt.set_title("目标 XY 轨迹（红=已走完的点）")
    axt.plot(xs, ys, "-", lw=0.6, color="0.7")
    tgt_pt, = axt.plot([], [], "o", ms=5, color="tab:blue", label="最新目标")
    done_pt, = axt.plot([], [], "o", ms=5, color="tab:red", label="臂已到")
    axt.set_aspect("equal")
    axt.legend(fontsize=8)

    def update(fi):
        t = times[fi]
        px, py, cols = [], [], []
        c = {s: 0 for s in STAGES}
        for r in rows:
            s = stage_of(r, t)
            if s is None:
                continue
            c[s] += 1
        for r in rows:
            s = stage_of(r, t)
            if s is None or s == "done":
                continue
            i = STAGES.index(s)
            k = c[s]
            px.append(i + (np.random.RandomState(r["pid"]).rand() - 0.5) * 0.5)
            py.append(min(10.5, 0.4 + 0.55 * (r["pid"] % 19)))
            cols.append({"frame": "0.6", "filt": "tab:green", "queue": "tab:orange",
                         "push": "tab:red"}[s])
        scat.set_offsets(np.c_[px, py] if px else np.empty((0, 2)))
        scat.set_color(cols if cols else "none")
        for i, s in enumerate(STAGES):
            counts_txt[i].set_text(f"{c[s]}")
        clock.set_text(f"t = {t:5.2f} s")
        for s in STAGES:
            lines[s].set_data(times[:fi + 1], cnt[s][:fi + 1])
        cur = [r for r in rows if r["t_queue"] is not None and r["t_queue"] <= t]
        if cur:
            tgt_pt.set_data([cur[-1]["x"]], [cur[-1]["y"]])
        dn = [r for r in rows if r["t_done"] is not None and r["t_done"] <= t and r["actual_x"] is not None]
        if dn:
            done_pt.set_data([dn[-1]["actual_x"]], [dn[-1]["actual_y"]])
        return [scat, *counts_txt, clock, *lines.values(), tgt_pt, done_pt]

    ani = FuncAnimation(fig, update, frames=len(times), interval=1000 / fps, blit=False)
    out = Path(out)
    try:
        ani.save(str(out), writer=FFMpegWriter(fps=fps, bitrate=2400))
    except Exception as e:  # noqa: BLE001
        out = out.with_suffix(".gif")
        print(f"  ffmpeg 不可用（{type(e).__name__}），改存 GIF")
        ani.save(str(out), writer=PillowWriter(fps=min(fps, 15)))
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="逐点 CSV → 表格 + 流水线动画")
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="", help="动画输出路径，默认与 CSV 同名 .mp4")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--speed", type=float, default=1.0, help=">1 快放")
    a = ap.parse_args()
    for path in a.csv:
        rows = load(path)
        if not rows:
            print(f"{path}: 空")
            continue
        table(rows, path)
        if not a.table_only:
            out = a.out or str(Path(path).with_suffix(".mp4"))
            got = animate(rows, out, a.fps, a.speed)
            print(f"  动画: {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
