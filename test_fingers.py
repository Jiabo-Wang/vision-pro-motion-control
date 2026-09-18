#!/usr/bin/env python3
"""逐根手指自检 —— 绕过全部跟踪逻辑，直接命令机械手。

用来把「小指/无名指不动」这类问题一刀切成两半：
  这里动  -> 机械手没问题，是**头显没跟踪到**那两根手指
  这里不动 -> 机械手/驱动的问题，跟头显无关

只动一根，其余保持不动（-1），所以不会出现两指对顶。
小指和无名指本来就碰不到拇指，这个测试很安全。

    python test_fingers.py               # 全部六维逐个测
    python test_fingers.py --dof 0 1     # 只测小指和无名指
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspire_hand6 import (ANGLE_HOLD, DOF_NAMES, NUM_DOF, InspireHand6,
                           decode_error)


def main() -> int:
    p = argparse.ArgumentParser(description="逐根手指自检")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--dof", type=int, nargs="*", default=list(range(NUM_DOF)),
                   help="要测哪几维，默认全部。0小指 1无名指 2中指 3食指 4拇指弯 5拇指旋")
    p.add_argument("--force", type=int, default=400)
    p.add_argument("--speed", type=int, default=500)
    p.add_argument("--settle", type=float, default=1.8, help="每步等多久让手指走到位")
    args = p.parse_args()

    hand = InspireHand6(port=args.port)
    hand.connect()
    hand.set_force(args.force)
    hand.set_speed(args.speed)
    print(f"已连接 {args.port}  力控 {args.force}g  速度 {args.speed}")
    print(f"起始角度: {hand.get_angles()}")

    # 先验活性：手可能处在「通信活着、控制循环死了」的半死状态，
    # 那种状态下逐维去动它纯属浪费 30 秒，而且结论会误导（看着像六维全坏）。
    ok, why = hand.liveness()
    if not ok:
        print("\n" + "!" * 62)
        print("!! 手的控制循环没在跑，下面的逐维测试没有意义。")
        print(f"!!   {why}")
        print("!! → **给手断电重启**：拔掉手自己的 24V 电源（不是 USB！），")
        print("!!   等 5 秒再上电。USB-CAN 适配器是独立供电的，拔它没用。")
        print("!! 重启成功的判据：温度读数会变（它一直在漂）。")
        print("!" * 62)
        hand.close()
        return 2
    print(f"活性检测: {why}\n")

    print(f"{'维':<14}{'指令':>6}{'实测前':>8}{'实测后':>8}{'移动':>8}   判定")
    print("-" * 62)
    bad = []
    try:
        for d in args.dof:
            if not 0 <= d < NUM_DOF:
                continue
            lo, hi = hand.limits[d]
            moved_total = 0
            for target in (hi, lo, hi):          # 张开 -> 闭合 -> 张开
                before = hand.get_angles()[d]
                six = [ANGLE_HOLD] * NUM_DOF     # 其余维保持不动
                six[d] = target
                hand.set_angles(six)
                time.sleep(args.settle)
                after = hand.get_angles()[d]
                moved = abs(after - before)
                moved_total += moved
                print(f"{DOF_NAMES[d]:<14}{target:>6}{before:>8}{after:>8}{moved:>8}")
            verdict = "✓ 会动" if moved_total > 100 else "✗ **不动**"
            if moved_total <= 100:
                bad.append(DOF_NAMES[d])
            print(f"{'':<14}{'':>6}{'':>8}{'':>8}{moved_total:>8}   {verdict}\n")

        print("=" * 62)
        if bad:
            print(f"这几维**机械手自己就不动**: {', '.join(bad)}")
            print("说明和头显跟踪无关，是手/驱动的问题。看下面的诊断：")
        else:
            print("六维都能动 —— 机械手没问题。")
            print("那「小指/无名指不跟踪」就是**头显没给出这两根手指的数据**，")
            print("用 avp_arm_teleop.py --hand-debug 看 `littl raw=` 会不会变。")
        print("\n手的状态：")
        print(hand.diagnose())
        print(f"\n实际受力: {hand.get_force()}")
    finally:
        try:
            hand.set_angles([1000] * NUM_DOF)    # 收尾张开，别让手夹着
        except Exception as e:
            print(f"张手失败: {type(e).__name__}: {e}")
        hand.close()


if __name__ == "__main__":
    sys.exit(main() or 0)
