#!/usr/bin/env python3
"""只连头显，量**数据本身**好不好。完全不碰机械臂。

用来回答「是不是传感器捕捉有问题」。臂在链路末端，它抖可能是自己的问题，
也可能是喂给它的数据本来就烂 —— 这个脚本把头显那一端单独摘出来量，
量完就知道该往哪边查。

三项，按提示做动作：

  1. **静止漂移/噪声**：手举着别动 10 秒。
     位置噪声应当是亚毫米级；如果有毫米级抖或者持续朝一个方向漂，
     那臂抖就是数据抖，往上游修（--arm-smooth），调臂没用。

  2. **帧到达节奏**：同时统计。到达间隔忽大忽小说明**网络在抖**，
     遥操会拿到成批/陈旧的帧 —— 表现也是臂抖，但滤波治不了。
     （本机实测 ping 头显 5~407ms，mdev 130ms，这条很值得看）

  3. **位移标度**：手平移一段**已知距离**（默认 30cm）再停住。
     头显报的位移和实际差太多，绝对映射就是错的 —— 这直接对应
     「人手增量和机械臂增量对不对得上」。

    python probe_hand_data.py <头显IP>
    python probe_hand_data.py --scan
    python probe_hand_data.py <头显IP> --move 0.20     # 改成量 20cm
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_axes import connect_streamer_guarded, scan_for_streamer  # noqa: E402


def wrist(d, side="left"):
    return np.asarray(d[f"{side}_wrist"]).reshape(4, 4)


def collect(s, seconds, log, label):
    """按帧收集，返回 (时间戳, 位置, 旋转矩阵)。只取**新**帧。"""
    ts, ps, Rs = [], [], []
    last = None
    t0 = time.time()
    while time.time() - t0 < seconds:
        d = s.get_latest()
        if d is None:
            time.sleep(0.002)
            continue
        T = wrist(d)
        if last is not None and np.array_equal(T, last):
            time.sleep(0.002)
            continue                      # 同一帧，不重复计数
        last = T
        ts.append(time.time())
        ps.append(T[:3, 3].copy())
        Rs.append(T[:3, :3].copy())
        left = seconds - (time.time() - t0)
        if len(ts) % 30 == 0:
            log(f"    {label} ... 还剩 {left:.0f}s（已收 {len(ts)} 帧）")
    return np.array(ts), np.array(ps), np.array(Rs)


def main() -> int:
    p = argparse.ArgumentParser(description="量头显数据质量（不碰机械臂）")
    p.add_argument("ip", nargs="?", help="头显 IP（不填就扫描）")
    p.add_argument("--scan", action="store_true")
    p.add_argument("--still", type=float, default=10.0, help="静止段秒数")
    p.add_argument("--move", type=float, default=0.30, help="平移段的已知距离(米)")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"（忽略用不到的参数: {' '.join(unknown)}）", flush=True)

    def log(m):
        print(m, flush=True)

    if args.scan or not args.ip:
        log("扫描 Tracking Streamer ...")
        found = scan_for_streamer(log=log)
        if not found:
            log("  没扫到。头显戴上、Tracking Streamer 点 Start、同一个 Wi-Fi。")
            return 1
        args.ip = found[0]
        log(f"  用 {args.ip}")

    s = connect_streamer_guarded(args.ip, log=log)
    if s is None:
        return 1

    # ── 1+2: 静止 ──────────────────────────────────────────────────────
    log(f"\n【1】把**左手**举在身前，**保持不动** {args.still:.0f} 秒。3 秒后开始 ...")
    time.sleep(3)
    ts, ps, Rs = collect(s, args.still, log, "静止")
    if len(ts) < 20:
        log(f"  只收到 {len(ts)} 帧，太少。手伸进头显视野里再试。")
        return 1

    dt = np.diff(ts)
    hz = 1.0 / np.median(dt)
    drift = np.linalg.norm(ps[-1] - ps[0]) * 1000
    noise = np.linalg.norm(ps - ps.mean(axis=0), axis=1)
    hf = np.abs(np.diff(ps, n=2, axis=0)).max(axis=1) * 1000

    log(f"\n  帧率 中位 {hz:5.1f} Hz   收到 {len(ts)} 帧")
    log(f"  到达间隔 中位 {np.median(dt)*1000:5.1f}ms  "
        f"p95 {np.percentile(dt,95)*1000:6.1f}ms  最大 {dt.max()*1000:7.1f}ms")
    if dt.max() > 0.15:
        log(f"    ⚠ 最大间隔 {dt.max()*1000:.0f}ms —— **网络/推流在卡**。"
            "遥操会拿到陈旧帧，表现成臂一顿一顿，滤波治不了这个。")
    log(f"  静止 {args.still:.0f}s 的净漂移 {drift:6.2f} mm"
        + ("   ⚠ 漂得明显，绝对映射会慢慢跑偏" if drift > 10 else "   ✓"))
    log(f"  位置噪声 RMS {noise.std()*1000:5.2f} mm   高频抖动 p95 "
        f"{np.percentile(hf,95):5.2f} mm"
        + ("   ⚠ 毫米级，臂抖多半是这里来的" if np.percentile(hf,95) > 1.0 else "   ✓"))

    # ── 3: 已知位移 ────────────────────────────────────────────────────
    log(f"\n【2】把左手朝一个方向**平移 {args.move*100:.0f} cm**（用尺子/桌沿比一下），"
        "然后停住。5 秒后开始记 8 秒 ...")
    time.sleep(5)
    _, ps2, _ = collect(s, 8.0, log, "平移")
    if len(ps2) < 20:
        log("  帧太少")
        return 1
    span = float(np.linalg.norm(ps2.max(axis=0) - ps2.min(axis=0)))
    log(f"\n  头显报告的位移 {span*100:6.2f} cm   你实际移动 {args.move*100:.0f} cm")
    ratio = span / max(args.move, 1e-9)
    log(f"  标度比 {ratio:.3f}"
        + ("   ✓ 对得上" if 0.9 <= ratio <= 1.1 else
           "   ⚠ **对不上** —— 头显的位移和真实距离差这么多，"
           "绝对映射从源头就是错的"))
    log("\n结论怎么用：")
    log("  噪声/漂移大 → 数据的问题，调 --arm-smooth，调臂没用")
    log("  间隔忽大忽小 → 网络的问题，查 Wi-Fi / 头显省电策略")
    log("  标度比不对 → 映射从源头错，--scale 补不回来（那是线性缩放，不是这回事）")
    log("  三项都正常 → 数据没问题，回去查臂那一侧")
    return 0


if __name__ == "__main__":
    sys.exit(main())
