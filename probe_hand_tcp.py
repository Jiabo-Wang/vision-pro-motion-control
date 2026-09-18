#!/usr/bin/env python3
"""从网口探测灵巧手（Modbus TCP），主要目的是把**触觉**拿到手。

为什么必须走网口：触觉寄存器在 3000~5123，而 CAN 扩展帧标识符里的寄存器
地址字段只有 12 位（上限 4095）——食指(4110+)、拇指(4480+)、掌心(4900+)
在 CAN 上**根本寻址不到**。手的规格是「Modbus TCP + CAN2.0」，
Modbus TCP 存在的意义就是这个。

前置：手的网口默认 192.168.11.210:6000，本机要在同网段有地址。
      新网口没有 IPv4 时先（需要 sudo，你自己跑）：
          sudo ip addr add 192.168.11.100/24 dev <网口名>

    python probe_hand_tcp.py                    # 用默认 IP
    python probe_hand_tcp.py --ip 192.168.11.210
"""
import argparse
import socket
import struct
import sys

# 已知寄存器（字节地址，和 CAN 那边同一套）
ANGLE_ACT, FORCE_ACT, TEMP = 1546, 1582, 1618
TACTILE = [(3000, 18, "小拇指指端"), (3018, 192, "小拇指指尖"),
           (4110, 18, "食指指端"), (4128, 192, "食指指尖"),
           (4900, 224, "掌心")]


def reachable(ip, port, timeout=2.0):
    s = socket.socket()
    s.settimeout(timeout)
    try:
        return s.connect_ex((ip, port)) == 0
    finally:
        s.close()


def main():
    p = argparse.ArgumentParser(description="灵巧手 Modbus TCP 探测")
    p.add_argument("--ip", default="192.168.11.210")
    p.add_argument("--port", type=int, default=6000)
    p.add_argument("--unit", type=int, default=1)
    args = p.parse_args()

    print(f"目标 {args.ip}:{args.port}")
    if not reachable(args.ip, args.port):
        print("  ✗ 端口连不上。检查：")
        print("    1) 本机在 192.168.11.x 网段上有地址吗？")
        print("       ip -br addr   →  新网口应当有 192.168.11.x")
        print("       没有的话:  sudo ip addr add 192.168.11.100/24 dev <网口名>")
        print("    2) 手的网线插在那个口上吗？(ip -br link 看 carrier)")
        print("    3) 手的 IP 是不是改过？默认 192.168.11.210")
        return 1
    print("  ✓ 端口通")

    from pymodbus.client.sync import ModbusTcpClient
    c = ModbusTcpClient(args.ip, port=args.port)
    if not c.connect():
        print("  ✗ Modbus 连接失败")
        return 1

    # 寄存器地址在 Modbus 上是按「字节」还是按「字」编址，各家实现不一样。
    # 拿 ANGLE_ACT 当试金石：它应该回 6 个 0..1000 的角度。
    print("\n判定寄存器编址方式（用 ANGLE_ACT=1546 当试金石，应回 6 个 0..1000）：")
    mode = None
    for name, addr in (("byte（地址直接用 1546）", ANGLE_ACT),
                       ("word（地址用 1546/2=773）", ANGLE_ACT // 2)):
        try:
            r = c.read_holding_registers(addr, 6, unit=args.unit)
            if r.isError():
                print(f"  {name}: 报错 {r}")
                continue
            vals = [v - 65536 if v > 32767 else v for v in r.registers]
            # 判据必须排掉「全 -1」：那是「该寄存器不支持」的典型返回，不是角度。
            # 第一版写的是 `-1 <= v <= 1000`，-1 正好卡在边界，六个 -1 被判成
            # 「✓ 像角度」—— 侥幸因为 byte 先试才没选错。要求：全在 0..1000，
            # 且至少有两个不同的值（六个一样的数也不像真实手指角度）。
            in_range = all(0 <= v <= 1000 for v in vals)
            varied = len(set(vals)) >= 2
            good = in_range and varied
            why = "" if good else ("（全 -1 = 不支持）" if all(v == -1 for v in vals)
                                   else "（超量程或六维全同）")
            print(f"  {name}: {vals}   {'✓ 像角度' if good else '✗ 不像' + why}")
            if good and mode is None:
                mode = "byte" if addr == ANGLE_ACT else "word"
        except Exception as e:
            print(f"  {name}: 异常 {type(e).__name__}: {e}")
    if mode is None:
        print("\n两种编址都没读出像样的角度，先别继续。把上面的输出发出来。")
        c.close()
        return 1
    print(f"\n编址方式: {mode}")

    def rd(byte_addr, n_bytes):
        a = byte_addr if mode == "byte" else byte_addr // 2
        r = c.read_holding_registers(a, n_bytes // 2, unit=args.unit)
        return None if r.isError() else r.registers

    print("\n── 触觉寄存器（CAN 上够不到的那些）──")
    for addr, n, label in TACTILE:
        probe = min(16, n)
        try:
            regs = rd(addr, probe)
            if regs is None:
                print(f"  {label:<12} @{addr:<5} 读失败")
                continue
            nz = sum(1 for v in regs if v)
            print(f"  {label:<12} @{addr:<5} 前{len(regs)}个字: {regs}  "
                  f"[{'有数据' if nz else '全 0'}]")
        except Exception as e:
            print(f"  {label:<12} @{addr:<5} 异常 {type(e).__name__}: {e}")

    print("\n注意：本机是 -T2（电容式 5 点触觉），不是 T1 的 17 点满配，")
    print("      所以只有一部分块会有数据，全 0 的块属正常。")
    c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
