#!/usr/bin/env python3
"""定 `--yaw`：看你的手往某个方向动时，臂**会**往哪动。**完全不碰机械臂。**

为什么需要这个：头显系和臂基座系之间差一个绕 Z 的旋转，这个角度取决于
你站的方位和臂的安装朝向——我在代码里看不到，只能你现场读一次。

而且它**同时决定平移和姿态**：`R_rel = R @ ΔR_手 @ Rᵀ`，转轴会被 R 一起搬走。
yaw 错 90° 的表现是：翻手掌（绕前臂滚转）落到臂的左右轴上，变成末端上下俯仰——
就是「手掌和手背识别不出来」。所以这一个数错了，位置和姿态会同时错。

    python probe_axes.py <头显IP>              # 扫所有候选 yaw
    python probe_axes.py <头显IP> --yaw -90    # 只验一个

用法：跑起来之后，**把左手朝一个明确方向平移 20cm 以上**（比如正右方），
看它打印的「臂会往 X」是不是你要的。四个候选里选那个全对的。

头显系（avp_stream 送出来的已经过 YUP2ZUP=Rx(-90°)）：**X=右, Y=前, Z=上**。
（README 一度写成「X 向前 Y 向左」，是错的，以 avp_stream/streamer.py 的
YUP2ZUP 为准。）
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from hand_retarget import palm_facing  # noqa: E402

CAND = (0, -90, 90, 180)
ARM = {(1, 0, 0): "前", (-1, 0, 0): "后", (0, 1, 0): "左",
       (0, -1, 0): "右", (0, 0, 1): "上", (0, 0, -1): "下"}
# 镜像候选：det=-1，**旋转怎么转都变不出来**。
# 如果四个纯 yaw 里没有一个处处对，那就是真镜像，必须用这一类。
# 头显系 X=右 Y=前：翻左右 = 翻 X，翻前后 = 翻 Y
MIRROR = {"mirror=lr(翻左右)": np.diag([-1.0, 1.0, 1.0]),
          "mirror=fb(翻前后)": np.diag([1.0, -1.0, 1.0])}


def connect_streamer_guarded(ip, port=12345, connect_timeout=6.0, data_timeout=5.0,
                             log=print):
    """连头显，**带超时和可诊断的失败**。

    直接 `VisionProStreamer(ip=...)` 在 IP 不通时会**无限期阻塞**，一个字都不打印，
    表现成「连不上、没反应」。而且 print 不带 flush 的话，管道/重定向下
    连已经打出来的话都看不见（块缓冲），更没法查。

    所以：先用普通 socket 确认端口通（2 秒内给结论），再把构造放到线程里跑，
    超时就明确报出来，而不是挂死。
    """
    import socket
    import threading

    ip = str(ip).strip()
    log(f"连接 {ip}:{port} ...")

    # 先挡住「根本不是个地址」的情况。`connect_ex` 在解析不了主机名时
    # **抛 socket.gaierror**（不是返回错误码），会直接崩成 traceback：
    #     socket.gaierror: [Errno -2] Name or service not known
    # 最常见的原因是把文档里的占位符 <头显IP> 原样敲了进去。
    # 最常见的手滑：IP 和后面的参数之间少个空格，粘成 `192.168.0.246--no-cameras`。
    # 单说「不是合法 IPv4」帮不上忙，得直接指出来。
    if "--" in ip:
        head = ip.split("--", 1)[0]
        log(f"  ✗ `{ip}` 里混进了参数 —— IP 和参数之间**少了一个空格**。")
        log(f"    你要的大概是:  {head} --{ip.split('--', 1)[1]}")
        return None
    if ip.startswith("<") or ip.endswith(">") or "头显" in ip or "IP" == ip.upper():
        log(f"  ✗ `{ip}` 是个占位符，不是地址。")
        log("    换成 Tracking Streamer 界面上显示的那个 IP，例如 192.168.0.4")
        return None
    try:
        socket.inet_aton(ip)
    except OSError:
        log(f"  ✗ `{ip}` 不是合法的 IPv4 地址。")
        log("    头显 IP 在 Tracking Streamer 应用界面上，形如 192.168.x.x，"
            "每次连 Wi-Fi 都可能变。")
        return None

    # 探测要**重试**。这台头显的 Wi-Fi 会瞬时掉线（实测 ping 抖动 5~407ms，
    # 偶尔单次 1 秒超时），单次探测就下结论会把「网络抖了一下」误判成
    # 「IP 不对」，然后把人指去查一个根本没问题的地方。
    rc, tries = None, 4
    for k in range(tries):
        sk = socket.socket()
        sk.settimeout(2.5)
        try:
            rc = sk.connect_ex((ip, port))
        except socket.gaierror as exc:
            log(f"  ✗ 解析不了 `{ip}`: {exc}")
            return None
        except OSError as exc:
            rc = -1
            log(f"  第 {k+1}/{tries} 次: {exc}")
        finally:
            sk.close()
        if rc == 0:
            break
        if rc in (111, 61):
            break                         # 明确拒绝，重试没意义
        if k < tries - 1:
            log(f"  第 {k+1}/{tries} 次没通（errno={rc}），1 秒后重试 ...")
            time.sleep(1.0)

    if rc != 0:
        # 主机在不在？同样要重试，理由同上。
        up = 0
        for _ in range(3):
            try:
                import subprocess
                if subprocess.run(["ping", "-c1", "-W2", ip],
                                  capture_output=True, timeout=5).returncode == 0:
                    up += 1
            except Exception:              # noqa: BLE001
                pass
        if rc in (111, 61) or up > 0:
            log(f"  ✗ {ip} **在线**（ping {up}/3 通），但 {port} 端口没人监听。")
            log("    → Tracking Streamer 没在推流。**IP 是对的，不用换。** 去头显里：")
            log("      1) 打开 Tracking Streamer 应用（不能只在后台）")
            log("      2) 点界面上的 **Start**")
            log("      3) 保持戴着 —— 摘下来会停止推流")
            if 0 < up < 3:
                log(f"    ⚠ 顺带：ping 只通了 {up}/3，这台头显的 Wi-Fi 在掉线，"
                    "遥操时会表现成臂一顿一顿")
        else:
            log(f"  ✗ {ip} 连续 {tries} 次都不可达，ping 也 0/3。")
            log("    → 这次才是真的 IP 不对 / 不在同一网络 / 头显休眠了。")
            log("    用 `python probe_axes.py --scan` 扫一遍。")
        return None

    log("  ✓ 端口通，正在建流 ...")

    box = {}

    def _mk():
        try:
            from avp_stream import VisionProStreamer
            box["s"] = VisionProStreamer(ip=ip, record=False)
        except Exception as exc:                      # noqa: BLE001
            box["e"] = exc

    th = threading.Thread(target=_mk, daemon=True)
    th.start()
    th.join(connect_timeout)
    if "e" in box:
        log(f"  ✗ 建流失败: {type(box['e']).__name__}: {box['e']}")
        return None
    if "s" not in box:
        log(f"  ✗ 建流超过 {connect_timeout:.0f}s 没返回（端口通但没有推流）。"
            "多半是 Tracking Streamer 没点 Start，或者头显没戴上。")
        return None
    s = box["s"]

    import time as _t
    t0 = _t.time()
    while _t.time() - t0 < data_timeout:
        if s.get_latest() is not None:
            log("  ✓ 已收到数据")
            return s
        _t.sleep(0.05)
    log(f"  ✗ 连上了但 {data_timeout:.0f}s 内没有任何数据帧。"
        "头显戴上、手伸进视野再试。")
    return None


def scan_for_streamer(port=12345, timeout=0.35, log=print):
    """在本机各个网段上扫一遍 12345 端口，找 Tracking Streamer。

    头显 IP 每次连 Wi-Fi 都可能变，而应用界面上的那个数又要摘下头显才看得到 ——
    扫一遍比来回折腾快。只探本机已经在的网段，不碰别人的网络。
    """
    import ipaddress
    import socket
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    nets, mine = [], set()
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr"], capture_output=True,
                             text=True, timeout=5).stdout
        for line in out.splitlines():
            parts = line.split()
            if "inet" in parts:
                cidr = parts[parts.index("inet") + 1]
                mine.add(cidr.split("/")[0])          # 本机自己的地址，不是头显
                net = ipaddress.ip_network(cidr, strict=False)
                if not net.is_loopback and net.num_addresses <= 4096:
                    nets.append(net)
    except Exception as exc:                       # noqa: BLE001
        log(f"  读网卡失败: {exc}")
        return []

    # 网段可能互相重叠（192.168.2.222 同时落在 /24 和 /21 里），去重；
    # 再把本机自己的地址剔掉 —— 扫到自己没有意义，只会让人以为找到了头显。
    hosts = sorted({str(h) for n in nets for h in n.hosts()} - mine)
    log(f"  扫 {len(nets)} 个网段 / {len(hosts)} 个地址的 {port} 端口 ...")

    def probe(h):
        sk = socket.socket()
        sk.settimeout(timeout)
        try:
            return h if sk.connect_ex((h, port)) == 0 else None
        except OSError:
            return None
        finally:
            sk.close()

    with ThreadPoolExecutor(max_workers=256) as ex:
        found = [h for h in ex.map(probe, hosts) if h]
    return found


def yaw_matrix(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def arm_dir(v):
    """把一个臂基座系的向量说成人话。假设臂基座 X=前 Y=左 Z=上。"""
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return "—"
    u = v / n
    k = int(np.argmax(np.abs(u)))
    axis = [0, 0, 0]
    axis[k] = 1 if u[k] > 0 else -1
    return ARM[tuple(axis)]


def hand_dir(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return "—"
    u = v / n
    k = int(np.argmax(np.abs(u)))
    return [("右", "左"), ("前", "后"), ("上", "下")][k][0 if u[k] > 0 else 1]


def main():
    p = argparse.ArgumentParser(description="定 --yaw，不碰机械臂")
    p.add_argument("ip", nargs="?", help="Vision Pro 的 IP（不填就自动扫描）")
    p.add_argument("--scan", action="store_true", help="扫本机各网段找 Tracking Streamer")
    p.add_argument("--yaw", type=float, default=None, help="只验这一个角度")
    p.add_argument("--min-move", type=float, default=0.12,
                   help="位移超过这么多米才判方向(默认 0.12)，太小的抖动不算")
    p.add_argument("--seconds", type=float, default=0.0)
    args, unknown = p.parse_known_args()
    if unknown:
        # 用户很自然会把主程序那条命令的参数一起复制过来。
        # 直接报 "unrecognized arguments" 然后退出太不友好，忽略并提示即可。
        print(f"（忽略了本脚本用不到的参数: {' '.join(unknown)}）", flush=True)

    def _log(m):
        print(m, flush=True)          # 管道下不 flush 就什么都看不见

    if args.scan or not args.ip:
        _log("扫描 Tracking Streamer ...")
        found = scan_for_streamer(log=_log)
        if not found:
            _log("  没扫到。确认头显戴着、Tracking Streamer 点了 Start、"
                 "而且和本机在同一个 Wi-Fi。")
            return 1
        _log(f"  找到: {', '.join(found)}")
        if not args.ip:
            args.ip = found[0]
            _log(f"  用 {args.ip}")

    s = connect_streamer_guarded(args.ip, log=_log)
    if s is None:
        return 1

    cands = CAND if args.yaw is None else (args.yaw,)
    print("\n把**左手**朝一个明确方向平移 20cm 以上，然后停住。Ctrl-C 退出。")
    print(f"候选 yaw: {', '.join(str(c)+'°' for c in cands)}")
    print("\n判据：")
    print("  1) 四个纯 yaw 里有一个**处处**和你想要的方向一致 → 用那个 --yaw。")
    print("  2) 一个都没有，但某个「镜像类」处处对 → 是**真镜像**，旋转修不了，")
    print("     告诉我，我加一个显式的反射映射（位置和姿态要一起反，否则手心手背会乱）。")
    print("  3) 中括号里的掌向应当跟你的手实际朝向一致 —— 这条验的是手心/手背辨识。\n")

    base = np.asarray(s.get_latest()["left_wrist"]).reshape(4, 4)[:3, 3].copy()
    t0 = time.time()
    last = ""
    try:
        while True:
            if args.seconds and time.time() - t0 > args.seconds:
                break
            d = s.get_latest()
            if d is None:
                time.sleep(0.05)
                continue
            p_now = np.asarray(d["left_wrist"]).reshape(4, 4)[:3, 3]
            delta = p_now - base
            dist = float(np.linalg.norm(delta))
            if dist < args.min_move:
                time.sleep(0.05)
                continue
            cols = "  ".join(f"yaw={c:>4.0f}°→**{arm_dir(yaw_matrix(c) @ delta)}**"
                             for c in cands)
            mir = "  ".join(f"{k}→**{arm_dir(M @ delta)}**" for k, M in MIRROR.items())
            try:
                facing = palm_facing(np.asarray(d["left_fingers"]),
                                     np.asarray(d["left_wrist"]).reshape(4, 4)[:3, :3])
            except Exception as exc:  # noqa: BLE001
                facing = f"掌向算不出({type(exc).__name__})"
            line = (f"手往 {hand_dir(delta):<2} ({dist*100:4.1f}cm)  [{facing}]\n"
                    f"      纯旋转: {cols}\n"
                    f"      镜像类: {mir}")
            if line != last:
                print(line, flush=True)
                last = line
            # 停住不动 0.8 秒就重新取基准，方便连着测下一个方向
            time.sleep(0.05)
            if float(np.linalg.norm(
                    np.asarray(s.get_latest()["left_wrist"]).reshape(4, 4)[:3, 3]
                    - p_now)) < 0.005:
                time.sleep(0.8)
                base = np.asarray(s.get_latest()["left_wrist"]
                                  ).reshape(4, 4)[:3, 3].copy()
                print("   (已重新取基准，可以测下一个方向)", flush=True)
                last = ""
    except KeyboardInterrupt:
        print("\n退出")
    print("\n选定之后加到主程序上，例如：")
    print("    python avp_arm_teleop.py <头显IP> --no-cameras --yaw -90")
    return 0


if __name__ == "__main__":
    sys.exit(main())
