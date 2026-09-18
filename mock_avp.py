#!/usr/bin/env python3
"""假 Tracking Streamer —— 没有头显时离线调 avp_arm_teleop 的参数用。

在 127.0.0.1:12345 上按真实协议推 30Hz 合成数据：
  左手  手腕沿 X 往复移动；手腕绕 Z 拧（转手掌）+ 绕 X 上下挥；五指周期张合
  右手  捏合距离在第 3 秒跨过阈值，用来驱动 dead-man

    python mock_avp.py &
    python avp_arm_teleop.py 127.0.0.1 --mock --mock-hand --no-cameras --auto-arm
"""
import math
import signal
import sys
import time
from concurrent import futures

import grpc
import numpy as np
from avp_stream.grpc_msg import handtracking_pb2 as pb
from avp_stream.grpc_msg import handtracking_pb2_grpc as pbg

PORT = 12345
# 关节编号同 avp_stream：拇指 1-4，食指 5-9，中指 10-14，无名 15-19，小指 20-24
CHAINS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8, 9), "middle": (10, 11, 12, 13, 14),
          "ring": (15, 16, 17, 18, 19), "little": (20, 21, 22, 23, 24)}
KNUCKLE_X = {"index": 0.00, "middle": -0.02, "ring": -0.04, "little": -0.06}


def mat(p, R=None):
    m = pb.Matrix4x4()
    R = np.eye(3) if R is None else R
    for i in range(3):
        for j in range(3):
            setattr(m, f"m{i}{j}", float(R[i, j]))
    m.m03, m.m13, m.m23 = float(p[0]), float(p[1]), float(p[2])
    return m


def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def hand_joints(curl: float, thumb_opp: float):
    """curl 0=完全张开 1=握拳，返回 25 个相对手腕的关节位置。"""
    J = np.zeros((25, 3))
    bend = curl * 1.15
    for name, x in KNUCKLE_X.items():
        ids = CHAINS[name]
        J[ids[0]] = [x, 0.02, 0.0]
        pos = np.array([x, 0.045, 0.0])
        J[ids[1]] = pos
        for k in range(1, 4):
            a = bend * k
            pos = pos + np.array([0.0, math.cos(a), -math.sin(a)]) * 0.030
            J[ids[1 + k]] = pos
    tdir = np.array([0.25 + 0.75 * thumb_opp, 1.0 - 0.6 * thumb_opp, 0.0])
    tdir /= np.linalg.norm(tdir)
    pos = np.array([0.025, 0.005, 0.0])
    J[1] = pos
    for k in range(1, 4):
        a = bend * k * 0.55
        Rm = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        pos = pos + Rm @ tdir * 0.028
        J[1 + k] = pos
    return J


def make_hand(wrist, curl, thumb_opp, pinch_gap=None, R=None):
    h = pb.Hand()
    h.wristMatrix.CopyFrom(mat(wrist, R))
    J = hand_joints(curl, thumb_opp)
    if pinch_gap is not None:              # 精确控制捏合距离时（右手 dead-man）
        J[4] = np.array([0.0, 0.0, 0.0])
        J[9] = np.array([pinch_gap, 0.0, 0.0])
    for j in range(25):
        h.skeleton.jointMatrices.append(mat(J[j]))
    return h


class Mock(pbg.HandTrackingServiceServicer):
    def StreamHandUpdates(self, request, context):
        print(f"[mock] 客户端连接 {context.peer()}", flush=True)
        t0, n = time.time(), 0
        while context.is_active():
            t = time.time() - t0
            u = pb.HandUpdate()
            u.Head.CopyFrom(mat((0.0, 0.0, 1.6)))
            curl = 0.5 - 0.5 * math.cos(t * 2 * math.pi / 6.0)
            # 绕 Z 拧 = 转手掌；绕 X = 上下挥。两个都做，方便区分 rot-mode
            Rw = rotz(0.5 * math.sin(t * 1.1)) @ rotx(0.5 * math.sin(t * 0.7))
            u.left_hand.CopyFrom(make_hand(
                (-0.30 + 0.08 * math.sin(t * 0.8), 0.0, 1.10), curl, thumb_opp=curl, R=Rw))
            u.right_hand.CopyFrom(make_hand((0.30, 0.0, 1.10), 0.2, 0.2,
                                            pinch_gap=0.050 if t < 3.0 else 0.010))
            yield u
            n += 1
            time.sleep(1 / 30)
        print(f"[mock] 推送 {n} 帧", flush=True)


def main():
    s = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pbg.add_HandTrackingServiceServicer_to_server(Mock(), s)
    s.add_insecure_port(f"127.0.0.1:{PORT}")
    s.start()
    print(f"[mock] 已启动 127.0.0.1:{PORT}（手腕会转、五指会张合）", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (s.stop(0), sys.exit(0)))
    s.wait_for_termination()


if __name__ == "__main__":
    main()
