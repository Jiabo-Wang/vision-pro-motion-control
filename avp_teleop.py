"""
VisionProTeleop 方案：透视模式下读取 Vision Pro 头部 / 手腕 / 手指位姿，并给出机械臂遥操作骨架。

与 Vuer 方案的区别
------------------
- Vision Pro 上运行 App Store 的 "Tracking Streamer" 应用（Improbable AI），透视模式，能看到真实机械臂。
- 数据流方向反过来：Vision Pro 是 gRPC 服务器（端口 12345），本脚本是客户端，主动连它的 IP。
- 坐标系：avp_stream 已把 ARKit 的 Y-up 转成 Z-up（右手系，单位米），与多数机械臂一致。
  原点是应用启动时头显所在位置，X 向前、Y 向左、Z 向上。

使用
----
    pip install avp_stream numpy
    1. Vision Pro 打开 Tracking Streamer，点 Start，记下界面上显示的 IP。
    2. 运行  python avp_teleop.py 192.168.3.100
       可选参数：--hz 30  --print-hz 5  --log avp_log.jsonl   （python avp_teleop.py -h 查看）
    3. 电脑和 Vision Pro 需在同一局域网，Windows 防火墙放行 Python。

数据格式（data = streamer.get_latest()）
--------------------------------------
    data["head"]            (1,4,4)   头部位姿，世界系
    data["left_wrist"]      (1,4,4)   左手腕位姿，世界系
    data["right_wrist"]     (1,4,4)
    data["left_fingers"]    (25,4,4)  25 个关节，相对手腕系；世界系 = wrist @ finger
    data["right_fingers"]   (25,4,4)
    data["left_pinch_distance"]  float  拇指尖到食指尖距离（米），< 0.02 左右视为捏合
    data["right_pinch_distance"] float
    data["right_wrist_roll"]     float  手腕滚转角
关节索引：0 wrist, 4 thumbTip, 9 indexTip, 14 middleTip, 19 ringTip, 24 littleTip
"""

import argparse
import json
import time

import numpy as np
from avp_stream import VisionProStreamer

# ============ 配置 ============
# Vision Pro IP / 控制频率 / 打印频率 / 日志文件 由命令行参数指定，见 parse_args()

PINCH_ON = 0.020             # 捏合判定阈值（米），小于此值算捏住
PINCH_OFF = 0.035            # 松开阈值，做滞回避免抖动
GRIPPER_OPEN_DIST = 0.08     # 捏合距离 >= 该值时夹爪全开
POS_SCALE = 1.0              # 手移动 1 m，机械臂末端移动 POS_SCALE m
MAX_STEP = 0.02              # 每个控制周期末端最多移动多少米（限速，安全）
SMOOTH = 0.3                 # 一阶低通系数，0 不平滑，越大越平滑（但更滞后）
# ==============================

JOINT = dict(wrist=0, thumbTip=4, indexTip=9, middleTip=14, ringTip=19, littleTip=24)


# ---------------- 数学工具 ----------------
def mat_to_pos_quat(T):
    """4x4 -> (xyz, 四元数 wxyz)。"""
    pos = T[:3, 3].copy()
    R = T[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return pos, np.array(q)


def finger_world(wrist_T, fingers, idx):
    """把相对手腕的关节矩阵变换到世界系。"""
    return wrist_T @ fingers[idx]


# ---------------- 机械臂接口（按你的机械臂改这里） ----------------
class RobotInterface:
    """
    把这几个方法换成你机械臂的 SDK / ROS 调用即可。
    位姿都在机械臂基座坐标系下，位置单位米，四元数 wxyz。
    """

    def __init__(self):
        # 初始末端位姿：真实机械臂请从 SDK 读取当前位姿
        self.ee_pos = np.array([0.4, 0.0, 0.3])
        self.ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.gripper = 1.0  # 1 全开，0 全闭

    def get_ee_pose(self):
        return self.ee_pos.copy(), self.ee_quat.copy()

    def send_ee_pose(self, pos, quat):
        # TODO: 换成 servo / cartesian move 指令
        self.ee_pos, self.ee_quat = pos, quat

    def send_gripper(self, opening):
        # TODO: 换成夹爪指令，opening ∈ [0,1]
        self.gripper = opening


# ---------------- 遥操作逻辑 ----------------
class Teleop:
    """
    离合式相对映射：
      - 右手捏合的瞬间，记录手腕位姿 H0 和末端位姿 E0；
      - 保持捏合期间，末端目标 = E0 + POS_SCALE * (H - H0)，姿态按相对旋转叠加；
      - 松开后机械臂停在原地，手可以自由移动“归位”，类似鼠标抬起再放下。
      - 左手捏合距离控制夹爪开合。
    这样不需要标定人手坐标系和机械臂坐标系的绝对关系。
    """

    def __init__(self, robot: RobotInterface):
        self.robot = robot
        self.engaged = False
        self.H0 = None
        self.E0_pos = None
        self.E0_R = None
        self.target_pos = None
        self.target_R = None

    @staticmethod
    def _pinch_state(prev, dist):
        if prev:
            return dist < PINCH_OFF
        return dist < PINCH_ON

    def step(self, data):
        right_wrist = np.asarray(data["right_wrist"])[0]      # (4,4)
        r_pinch = float(data["right_pinch_distance"])
        l_pinch = float(data["left_pinch_distance"])

        # ---- 右手：离合 + 末端跟随 ----
        engaged_now = self._pinch_state(self.engaged, r_pinch)
        if engaged_now and not self.engaged:
            # 刚捏住：记录参考
            self.H0 = right_wrist.copy()
            self.E0_pos, q = self.robot.get_ee_pose()
            self.E0_R = quat_to_mat(q)
            self.target_pos, self.target_R = self.E0_pos.copy(), self.E0_R.copy()
        self.engaged = engaged_now

        if self.engaged:
            dp = (right_wrist[:3, 3] - self.H0[:3, 3]) * POS_SCALE
            dR = right_wrist[:3, :3] @ self.H0[:3, :3].T
            raw_pos = self.E0_pos + dp
            raw_R = dR @ self.E0_R

            # 平滑
            self.target_pos = SMOOTH * self.target_pos + (1 - SMOOTH) * raw_pos
            self.target_R = raw_R  # 姿态直接跟随，可按需做 slerp

            # 限速
            cur_pos, _ = self.robot.get_ee_pose()
            step = self.target_pos - cur_pos
            n = np.linalg.norm(step)
            if n > MAX_STEP:
                step = step / n * MAX_STEP
            self.robot.send_ee_pose(cur_pos + step, mat_to_quat(self.target_R))

        # ---- 左手：夹爪 ----
        opening = np.clip((l_pinch - PINCH_ON) / (GRIPPER_OPEN_DIST - PINCH_ON), 0.0, 1.0)
        self.robot.send_gripper(float(opening))


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R):
    T = np.eye(4)
    T[:3, :3] = R
    return mat_to_pos_quat(T)[1]


# ---------------- 主循环 ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="通过 VisionProTeleop (Tracking Streamer) 读取 Vision Pro 位姿并遥操作机械臂",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("ip", help="Vision Pro 局域网 IP（Tracking Streamer 界面显示），如 192.168.3.100")
    p.add_argument("--hz", type=float, default=30, help="控制循环频率")
    p.add_argument("--print-hz", type=float, default=5, help="终端打印频率")
    p.add_argument("--log", default=None, metavar="FILE", help="把原始位姿写入 jsonl 文件，如 avp_log.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"连接 Vision Pro {args.ip}:12345 ...（请确认 Tracking Streamer 已点 Start）")
    streamer = VisionProStreamer(ip=args.ip, record=False)
    robot = RobotInterface()
    teleop = Teleop(robot)
    log_fp = open(args.log, "a", encoding="utf-8") if args.log else None

    period = 1.0 / args.hz
    print_period = 1.0 / args.print_hz
    last_print = 0.0
    n = 0
    while True:
        t0 = time.time()
        data = streamer.get_latest()
        if data is None:
            time.sleep(0.05)
            continue
        n += 1

        teleop.step(data)

        if log_fp:
            log_fp.write(json.dumps({
                "t": t0,
                "head": np.asarray(data["head"])[0].tolist(),
                "left_wrist": np.asarray(data["left_wrist"])[0].tolist(),
                "right_wrist": np.asarray(data["right_wrist"])[0].tolist(),
                "left_fingers": np.asarray(data["left_fingers"]).tolist(),
                "right_fingers": np.asarray(data["right_fingers"]).tolist(),
                "left_pinch": float(data["left_pinch_distance"]),
                "right_pinch": float(data["right_pinch_distance"]),
            }) + "\n")

        if t0 - last_print >= print_period:
            last_print = t0
            head = np.asarray(data["head"])[0]
            rw = np.asarray(data["right_wrist"])[0]
            lw = np.asarray(data["left_wrist"])[0]
            r_tip = finger_world(rw, np.asarray(data["right_fingers"]), JOINT["indexTip"])
            hp, hq = mat_to_pos_quat(head)
            rp, _ = mat_to_pos_quat(rw)
            lp, _ = mat_to_pos_quat(lw)
            tp, _ = mat_to_pos_quat(r_tip)
            ee_p, _ = robot.get_ee_pose()
            print(
                f"[HEAD ] pos=({hp[0]:+.3f},{hp[1]:+.3f},{hp[2]:+.3f}) quat=({hq[0]:+.2f},{hq[1]:+.2f},{hq[2]:+.2f},{hq[3]:+.2f})\n"
                f"[R-WRIST] ({rp[0]:+.3f},{rp[1]:+.3f},{rp[2]:+.3f}) indexTip=({tp[0]:+.3f},{tp[1]:+.3f},{tp[2]:+.3f}) "
                f"pinch={data['right_pinch_distance']:.3f}m engaged={teleop.engaged}\n"
                f"[L-WRIST] ({lp[0]:+.3f},{lp[1]:+.3f},{lp[2]:+.3f}) pinch={data['left_pinch_distance']:.3f}m gripper={robot.gripper:.2f}\n"
                f"[ROBOT] ee=({ee_p[0]:+.3f},{ee_p[1]:+.3f},{ee_p[2]:+.3f})  n={n}\n"
            )

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


if __name__ == "__main__":
    main()
