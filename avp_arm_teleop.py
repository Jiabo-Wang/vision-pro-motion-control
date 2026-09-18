#!/usr/bin/env python3
"""Vision Pro 左手 -> 珞石 AR5 + 因时 RH56E2（六指关节全跟踪）。

左手一只手管两件事：
    手腕位姿  -> 臂末端位姿（**绝对映射**：手什么姿态，末端就什么姿态）
    五根手指  -> 灵巧手 6 个自由度
右手只做一件事：**捏合 = dead-man**，松开臂立刻停。

跟踪方式：绝对伺服，不是增量累加
--------------------------------
接合瞬间记下「手的位姿」和「末端的位姿」作为基准，之后每帧算出**绝对目标位姿**，
把「目标 - 当前指令」作为这一帧下发。

为什么不能用增量：增量一旦被吃掉就永远找不回来 —— max_step_rotation 每步只放
0.1rad、被 IK/限位拒绝的那步整帧作废、自救重锚后对应关系断掉。仿真实测（前 4 秒
手快速乱转超出臂跟随能力、15% 丢帧，第 4 秒起手定住）：
    增量累加  手停后偏差永远停在 34.3°，再也回不去
    绝对伺服  手一停就追回来，偏差归 0.0°
后者被钳位只是「慢一点到位」，剩余误差下一帧继续发。

分工
----
臂走 `AR5Env.step()`（use_hand=False），沿用 vel_ar5 那套已验证的
delta 钳位 / 工作空间钳位 / 可达性闸 / 地板检查 / latch。
手走本目录自己的 `inspire_hand6.py`，跟 VEL 是两个项目、不共享代码。
两者各占各的资源：臂在网口，手在 /dev/ttyUSB0，不打架。

    # 0) 先标定（每个人手不一样，必做一次）
    python avp_arm_teleop.py <IP> --calib

    # 1) 完全不碰硬件：假臂 + 假手，只看数字
    python avp_arm_teleop.py <IP> --mock --mock-hand --no-cameras

    # 2) 只动真手，臂是假的
    python avp_arm_teleop.py <IP> --mock --real-hand --no-cameras
    #    注意必须带 --real-hand：光写 --mock 会把手一起关掉

    # 3) 真臂 + 真手（默认会先回起始姿态，腕关节从中位起步）
    python avp_arm_teleop.py <IP> --no-cameras

操作
----
  臂要动必须两道闸同时成立：键盘 ENTER 已挂离合 **且** 右手保持捏合。
  手**始终**跟随左手五指，与离合无关（方便先单独确认手的映射）。
  q 退出。
"""

import argparse
import re
import select
import threading
import sys
import termios
import time
from pathlib import Path

import logging

import numpy as np

AR5_ROOT = "/home/crp-5070ti-01/yuhang_workspace/vel_ar5/openvla-energy"
sys.path.insert(0, AR5_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from avp_stream import VisionProStreamer  # noqa: E402

from experiments.robot.ar5.ar5_env import AR5Env, ACTION_DIM  # noqa: E402
from experiments.robot.ar5.config import AR5Config  # noqa: E402
from experiments.robot.ar5.transforms import axis_angle_to_matrix  # noqa: E402

from hand_retarget import (Calibration, PINCH_FINGERS, Retargeter,  # noqa: E402
                           pinch_distances, raw_features)
from ar5_ik import AR5IK, AR5OptIK, ToolFrameKinematics  # noqa: E402
from vel_kin import matrix_to_rpy  # noqa: E402
from ar5_srs_ik import AR5SrsIK  # noqa: E402
from ar5_poe_ik import AR5PoeIK  # noqa: E402
from inspire_hand6 import DOF_NAMES, NUM_DOF, THUMB_ROT, InspireHand6  # noqa: E402

ERRLOG = None                                # 驱动最近的报错，见 LastErrors
PINCH_ON, PINCH_OFF = 0.020, 0.035          # 右手 dead-man 的滞回阈值
DEFAULT_CALIB = Path(__file__).resolve().parent / "hand_calib.json"


def parse_args():
    p = argparse.ArgumentParser(description="Vision Pro 左手驱动 AR5 + 灵巧手",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("ip", help="Vision Pro 的 IP")
    p.add_argument("--hz", type=float, default=50)
    p.add_argument("--print-hz", type=float, default=5)
    p.add_argument("--scale", type=float, default=1.0, help="手移动 1m，末端移动 scale m")
    p.add_argument("--yaw", type=float, default=0.0, help="头显坐标系绕 Z 转多少度再喂给臂")
    p.add_argument("--mirror", choices=("none", "lr", "fb"), default="none",
                   help="在 yaw 之前先做一次**反射**。镜像的行列式是 -1，"
                        "旋转怎么取都变不出来 —— 「左右反了但前后是对的」只能靠这个修。"
                        "lr=翻左右(头显系X), fb=翻前后(头显系Y)。"
                        "先用 probe_axes.py 确认到底是旋转还是镜像")
    p.add_argument("--no-rotation", dest="with_rotation", action="store_false",
                   help="只跟位置，不跟姿态。默认是**跟姿态**的——手什么姿态末端就什么姿态")
    p.add_argument("--rot-scale", type=float, default=1.0, help="姿态跟随的倍率")
    p.add_argument("--gain", type=float, default=0.5,
                   help="绝对伺服的比例增益。1.0=每帧把全部误差一次性下发（到钳位就是全速冲，顿挫）；小一点是指数逼近，丝滑但略滞后")
    p.add_argument("--tcp-z", type=float, default=0.0,
                   help="工具中心离法兰多远(米)，沿法兰自身 Z 轴。**默认 0 = 受控点就是法兰**。"
                        "手是通过定制法兰装上去的，长度和安装姿态都没实测过，"
                        "猜一个只会把误差变成有方向的系统偏差 —— 旋转中心会错在一个"
                        "说不清的位置上。量准了再填")
    p.add_argument("--payload", type=float, default=0.0,
                   help="法兰之后挂的质量(kg)。**默认 0=不设**：官方 790g 但实测外力约 1.41kg，差额是转接板+线缆。填错会让控制器把残余重力矩当成碰撞而保护停机。测准了再填")
    p.add_argument("--payload-com-z", type=float, default=0.075,
                   help="质心离法兰多远(米)。质量集中在掌部驱动块，不是指尖")
    p.add_argument("--no-estop-recover", action="store_true",
                   help="不要自动清急停。默认会清——这台臂没有物理急停按钮，只能走 recoverState(1) API")
    p.add_argument("--ik", choices=("opt", "poe", "srs", "dls", "stock"), default="opt",
                   help="逆解方式。opt=优化式（默认，chf_ws 路线：位姿误差当代价、限位和步长当硬约束，"
                        "永远给得出解）；dls=雅可比迭代改进版（自适应阻尼+WLN+早退）；stock=vel_ar5 原版。"
                        "poe=**旋量法解析解**（PoE + Paden-Kahan 子问题；需要 ar5_dh.json。残差 0.0000mm、1.9ms，比数值解更快更准；够不到时自动退数值解）。"
                        "srs=按 DH 硬推的闭式解（已被 poe 取代）。"
                        "基准（含驱动的 limits/jump 两道闸）：驱动接受率 dls 13.9%% → opt 100%%")
    p.add_argument("--stock-ik", action="store_true", help=argparse.SUPPRESS)  # 旧名，等价 --ik stock
    p.add_argument("--arm-smooth", type=float, default=1.0,
                   help="臂输入的 One-Euro 最小截止频率(Hz)。**越小越不抖但越滞后**。"
                        "0=完全关掉滤波（就是之前的行为）。臂抖就往小调，例如 0.6")
    p.add_argument("--arm-smooth-beta", type=float, default=3.0,
                   help="One-Euro 的速度系数。越大=快速运动时越跟手（但抖动也放得越开）")
    p.add_argument("--jitter-probe", action="store_true",
                   help="逐级量抖动来源：头显原始 → 滤波后 → IK 关节指令 → 实测关节。"
                        "退出时打印每一级的高频能量，抖在哪一级一眼看出来")
    p.add_argument("--lock-j5", type=float, default=None, metavar="RAD",
                   help="把 j5 锁死在这个角度(弧度)，只用其余 6 轴解末端位姿。"
                        "j5 是 ±50° 的窄轴之一，锁掉它等于把最容易顶限位的轴拿出问题，"
                        "同时消掉 7 轴的冗余（肘不会再自己换边）。不填=不锁")
    p.add_argument("--ik-max-step", type=float, default=0.0,
                   help="逆解每帧每关节最多走多少弧度。**0=自动**，取 max_joint_speed/hz "
                        "（臂一帧真正走得动的量）。原来固定 0.12，是臂实际速度的 8 倍，"
                        "指令一直跑在臂前面跳")
    p.add_argument("--arm-backend", choices=("rt", "movel"), default="rt",
                   help="臂怎么驱动。rt=实时关节流（本地旋量逆解，1.93ms，覆盖语义）；"
                        "movel=**MoveL 多路点流式**（本地只算绝对位姿，逆解交控制器）。"
                        "movel 实测(250mm/s 走走停停)：延迟 40ms、幅度不衰减、"
                        "append/moveStart 零错误，但队列语义下手停住时要吃一次 42.8ms 的 moveStart")
    p.add_argument("--movel-speed", type=float, default=250.0,
                   help="movel 后端的末端线速度 mm/s。**延迟 = 前导距离/速度**。"
                        "走走停停实测(真实遥操的样子)：150→66ms，**250→40ms**。"
                        "400 也是 ~41ms 但现场反馈「太快了」，而 250 同样达标，所以取 250。"
                        "注意：提速度比降预算管用——降预算只会让跳帧率飙升而延迟降不下来")
    p.add_argument("--movel-lead-ms", type=float, default=50.0,
                   help="movel 后端的延迟预算(ms)。换算成前导距离上限 = 速度×预算，"
                        "超了就跳过该帧（目标是最新值，不是必须执行的轨迹）。"
                        "**别单独调小这个**——会让跳帧率暴涨而延迟降不下来，先提 --movel-speed")
    p.add_argument("--movel-zone", type=float, default=3.0,
                   help="转弯区 mm，相邻 MoveL 段混合，避免一段一停")
    p.add_argument("--movel-deadband", type=float, default=3.0,
                   help="movel 的**抖动死区** mm：目标相对上一个已下发点动得比这还少，"
                        "就不下发。**延迟最有效的旋钮**：0.82mm 残余手抖下实测"
                        "0.5mm→73ms、3.0mm→**40ms**，而且跟踪 RMS 从 19.7 降到 14.2mm。"
                        "原理是短段加速度受限、臂跑不到指令速度（抖动路径上实测只有"
                        "25~33mm/s），死区把段拉长臂才提得起速。"
                        "代价是位置量化到 3mm，精细抓取嫌粗就降到 1.0")
    p.add_argument("--delta-frame", choices=("base", "tool"), default="tool",
                   help="人手位移按哪个系施加到末端。**默认 tool=末端系**（现场确认的手感）："
                        "映射跟着末端一起转 —— 末端转 90°，你「往前推」的方向也跟着转 90°。"
                        "base=基座系（遥操通例）：末端怎么转，「往前推」都是基座 +X。"
                        "接合那一刻两者完全一致，差别只在末端转起来之后。"
                        "⚠ 这治不了「臂型和人不一样」，那是零空间的事（见 --lock-j5）")
    p.add_argument("--flange-frame", action="store_true",
                   help="退回**法兰系**遥操（改之前的行为）。默认走工具系：设了 --tcp-z 时，"
                        "位姿链整体搬到工具中心，手腕原地转时灵巧手才是原地转。"
                        "法兰系下原地俯仰 30° 会把手甩出 69mm")
    p.add_argument("--no-tcp", action="store_true", help="不设 TCP/负载，维持原来的全 0")
    p.add_argument("--pinch-gate", type=float, default=0.45,
                   help="捏合驱动只在「指尖距离归一化后小于这个比例」时生效。调大=更容易被判成在捏（手指容易恒弯），调小=要靠得很近才触发")
    p.add_argument("--hand-debug", action="store_true",
                   help="打印每根手指的原始特征和映射结果，用来查「单独弯某根手指没反应」")
    p.add_argument("--rot-mode", choices=("roll", "full"), default="full",
                   help="full=手什么姿态末端什么姿态（默认）；roll=只跟手掌的拧转，上下挥手不跟")
    p.add_argument("--no-home", action="store_true",
                   help="不要启动时回起始姿态。默认是回的——腕关节 j5/j6 只有 ±50° 行程，不从中位起步很快就撞限位")
    p.add_argument("--home-wait", type=float, default=3.0,
                   help="回位前留几秒给你 Ctrl+C 取消")
    p.add_argument("--recentre", action="store_true",
                   help="回位后再把工具移到工作空间盒中心")
    p.add_argument("--log-every", type=float, default=2.0,
                   help="同一类驱动日志每多少秒最多打一条，其余折叠计数")
    p.add_argument("--slip-after", type=int, default=8,
                   help="臂连续多少帧到不了目标就「打滑」——就地重锚基准。绝对伺服不打滑的话会抱着一个永远追不上的目标死磕，最后 latch")
    p.add_argument("--relimit-after", type=int, default=15,
                   help="连续多少次被关节限位拒绝后，自动重锚到实测位姿（破死循环）")
    p.add_argument("--warn-joint-margin", type=float, default=8.0,
                   help="关节离软限位小于这个角度(度)就在状态行告警")
    p.add_argument("--max-rot-lead", type=float, default=0.35,
                   help="单帧最多下发多少弧度的姿态误差（≈20°）。绝对伺服下剩余误差下帧继续追，不会丢")
    p.add_argument("--max-lead", type=float, default=0.04,
                   help="指令位姿允许领先实测多少米。vel_ar5 默认 0.015(15mm)，对遥操太紧——手动得快一点指令就被拉回来，感觉像臂不跟。0.04 是放宽后的值")
    p.add_argument("--arm-speed", type=float, default=0.0,
                   help="臂的关节速度上限 rad/s。0=用 vel_ar5 的 0.74(42°/s，很保守)。想更跟手就往上给，1.2 左右是个起点。**这是真的让臂更快，先在小幅度动作上确认**")
    p.add_argument("--max-stale", type=float, default=0.25, help="多久没新帧就判链路陈旧")
    p.add_argument("--smooth", type=float, default=0.20, help="手指角度一阶低通系数")
    p.add_argument("--hand-hz", type=float, default=40, help="灵巧手下发频率（CAN 带宽有限）")
    p.add_argument("--hand-port", default="/dev/ttyUSB0")
    p.add_argument("--hand-force", type=int, default=400, help="力控阈值 g，全指跟踪时别给太大")
    p.add_argument("--hand-speed", type=int, default=1000,
                   help="灵巧手电机速度 0..1000。**原来默认 500 = 半速**，这是「手指跟踪慢」的主因。力控阈值才是保护，速度只影响快慢")
    p.add_argument("--freeze-thumb-rot", action="store_true",
                   help="拇指旋转锁在安全带中点不跟踪。这一维的原始特征不一定单调，抖就锁上")
    p.add_argument("--calib", action="store_true", help="进入标定流程并写入标定文件")
    p.add_argument("--calib-file", default=str(DEFAULT_CALIB))
    p.add_argument("--mock", action="store_true", help="假臂")
    p.add_argument("--real-hand", action="store_true", help="假臂配真手")
    p.add_argument("--no-hand", action="store_true", help="不接灵巧手")
    p.add_argument("--mock-hand", action="store_true",
                   help="用假串口跑手的整条链路（重定向/钳位/下发），不碰真手")
    p.add_argument("--no-cameras", action="store_true")
    p.add_argument("--auto-arm", action="store_true", help="跳过键盘离合，**只允许配 --mock**")
    p.add_argument("--seconds", type=float, default=0.0, help="跑满这么多秒自动退出")
    a = p.parse_args()
    if a.auto_arm and not a.mock:
        p.error("--auto-arm 只能和 --mock 一起用：真臂必须人工挂离合")
    return a


def yaw_matrix(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class OneEuro:
    """One-Euro 滤波器（Casiez et al. 2012）。给人手动捕数据去抖用的。

    为什么不用普通一阶低通：低通要压住抖就得把截止频率调低，而截止频率一低，
    快速运动就跟不上——遥操里表现成「不抖了但是发糊、手停了臂还在飘」。
    One-Euro 的截止频率**跟着速度走**：
        fc = min_cutoff + beta·|速度|
    人手停住时速度≈0 → 截止频率很低 → 抖动被压死；
    快速挥动时速度大 → 截止频率升上去 → 几乎不引入延迟。

    臂这条路上原来**一点滤波都没有**（`--smooth` 只喂给手的重定向），
    头显的手腕位姿原样进 IK，噪声就直接变成关节指令 —— 这是「臂很抖」的来源。
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 3.0,
                 d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x = None
        self._dx = None
        self._t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=float)
        if self._x is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        if not (1e-6 < dt < 1.0):          # 卡顿/回绕：重置，别拿脏 dt 算速度
            self._t = t
            return self._x.copy()
        self._t = t
        dx = (x - self._x) / dt
        self._dx += self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        self._x += self._alpha(cutoff, dt) * (x - self._x)
        return self._x.copy()


def filter_rotation(filt: OneEuro, R: np.ndarray, t: float) -> np.ndarray:
    """对旋转滤波：滤 6 维的两列，再正交化。

    不能直接对 9 个元素滤完就用 —— 滤出来的一般不是正交阵，
    送给 IK 会变成一个「带缩放的伪旋转」。取前两列滤，再 Gram-Schmidt 重建。
    """
    v = filt(np.concatenate([R[:, 0], R[:, 1]]), t)
    x, y = v[:3], v[3:]
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        return R
    x = x / nx
    y = y - x * float(y @ x)
    ny = np.linalg.norm(y)
    if ny < 1e-9:
        return R
    y = y / ny
    return np.column_stack([x, y, np.cross(x, y)])


def _auto_step(config, args) -> float:
    """每帧允许的关节步长。默认取臂**一帧真正走得动的量**。

    原来钉死 0.12 rad @50Hz = 6 rad/s，而 `config.arm.max_joint_speed` 是
    0.74 rad/s —— 指令比臂快 8 倍。后果不是臂走得快，而是**指令一直跑在臂前面**，
    每帧都在一个臂够不到的位置上重新解，指令序列自己就是抖的
    （仿真：单帧最大变化 1.38° → 0.85°，指令抖动 p95 0.135° → 0.078°）。
    """
    if args.ik_max_step and args.ik_max_step > 0:
        return float(args.ik_max_step)
    v = float(getattr(config.arm, "max_joint_speed", 0.74))
    return max(v / max(float(args.hz), 1.0), 1e-3)


def mirror_matrix(which: str) -> np.ndarray:
    """反射矩阵。det = -1，**任何旋转都变不出来**。

    「左右反了，但前后是对的」在数学上就不是旋转，是反射——所以之前调 --yaw
    一直调不好是必然的：yaw 转 180° 会把前后**一起**翻掉。

    和 yaw 合成之后 A = R_yaw @ M：
      平移  p → A p
      姿态  R_rel = A ΔR Aᵀ     （A 仍是正交阵，Aᵀ = A⁻¹）
    det(A) = -1，但 det(A ΔR Aᵀ) = det(ΔR) = +1 —— 共轭出来**仍是正经旋转**，
    不会给机械臂送一个带反射的姿态（那是无解的）。
    物理含义：臂看到的是你手部动作的镜像，手转的方向也跟着镜像。
    """
    # 头显系 X=右 Y=前，所以翻左右是翻 X、翻前后是翻 Y。
    # 参数名直接用 lr/fb，不用 x/y —— 第一版就是按轴名写的，自检里
    # 「镜像Y」实际翻的是前后，标签和行为正好对反，会把人指到错的选项上。
    return {"none": np.eye(3),
            "lr": np.diag([-1.0, 1.0, 1.0]),
            "fb": np.diag([1.0, -1.0, 1.0])}[which]


def left_fingers(data):
    return np.asarray(data["left_fingers"])


def wrist_xyz(data, side="left"):
    return np.asarray(data[f"{side}_wrist"]).reshape(4, 4)[:3, 3].copy()


def wrist_R(data, side="left"):
    return np.asarray(data[f"{side}_wrist"]).reshape(4, 4)[:3, :3].copy()


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 轴角向量（模长 = 转角）。

    `AR5Env.step` 的 delta[3:6] 就是轴角（transforms.compose_pose 用
    axis_angle_to_matrix 解它），**不是欧拉角**。
    早先这里写的是欧拉角相减，既类型不对、又会在 ±180° 附近
    把 +2° 算成 -358°，姿态跟随一开就会乱转。
    """
    tr = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(tr))
    if theta < 1e-9:
        return np.zeros(3)
    s = np.sin(theta)
    if abs(s) < 1e-9:          # 接近 180°，单步 delta 实际到不了这里
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2.0 * s) * theta


class LastErrors(logging.Handler):
    """记住驱动最近的 WARNING/ERROR，出问题时直接摆到人眼前。

    驱动把 RT 启动失败的真实原因放在一条 warning 里，而那条 warning 会被
    后面的输出刷掉；我原来的提示框只列了几种「常见原因」，恰好没有急停，
    结果最关键的线索反而被淹掉了。真实原因永远比猜测列表有用。
    """

    def __init__(self, keep=8):
        super().__init__(level=logging.WARNING)
        self.keep = keep
        self.records = []

    def emit(self, record):
        try:
            self.records.append(record.getMessage())
            del self.records[:-self.keep]
        except Exception:
            pass

    def matching(self, *words):
        return [m for m in self.records if any(w in m for w in words)]


# 报错关键词 -> 该怎么办。驱动的报错是中文原文，直接匹配。
FIX_HINTS = [
    (("急停", "estop", "E-stop"),
     "**急停仍然锁着**，而且 recoverState(1) 自动复位没能清掉它（上面有结果）。\n"
     "     这说明急停回路是**物理断开**的，不是控制器里一个可以软件清的锁存位：\n"
     "       · 有物理急停按钮的话把它旋起来\n"
     "       · 检查安全回路接线/门锁/安全继电器有没有断\n"
     "       · 示教器上复位并重新使能\n"
     "     急停是安全回路直接切断，和碰撞保护不是一回事，软件绕不过去。"),
    (("未上电", "power", "上电失败"),
     "臂没上电。检查使能开关/示教器上的上电状态。"),
    (("1338", "UDP"),
     "RT 的 UDP socket 打不开，多半是 local_ip 不对。\n"
     "     当前 config 里 local_ip 是 %s，确认本机在臂那个网段上的地址就是它。"),
    (("运动控制模式", "262"),
     "控制器卡在 RT 模式了（上次崩溃残留）。本脚本已尝试还原，重跑一次。"),
    (("碰撞", "collided"),
     "控制器判定碰撞。臂没真撞上的话多半是负载/质心填错，用 --payload 0 对比。"),
]


class RateLimit(logging.Filter):
    """同一类日志每 N 秒最多放一条，后面的折叠成计数。

    臂的 `IK failed` / `goal outside joint limits` 是每帧一条、50Hz 刷，
    状态行完全被冲掉，真正有用的信息（打滑、latch、关节余量）反而看不见。
    归类时**先把数字抹掉**：`IK failed for [0.4000,...]` 和
    `IK failed for [0.4001,...]` 是同一类。第一版按前 40 字符归类，
    结果坐标就在那 40 字符里，每条都是新 key，限流完全没生效。
    """

    _NUM = re.compile(r"[-+0-9.]+")

    def __init__(self, period=2.0):
        super().__init__()
        self.period = period
        self._last = {}
        self._held = {}

    def filter(self, record):
        key = self._NUM.sub("#", record.getMessage())[:80]
        now = time.time()
        last = self._last.get(key, 0.0)
        if now - last >= self.period:
            n = self._held.pop(key, 0)
            if n:
                record.msg = f"{record.getMessage()}   (同类已折叠 {n} 条)"
                record.args = ()
            self._last[key] = now
            return True
        self._held[key] = self._held.get(key, 0) + 1
        return False


class HandTracker(threading.Thread):
    """手的跟踪跑在独立线程里，**不受臂的循环拖累**。

    为什么必须拆出来：臂那边 IK 解不出来时会跑满 100 次迭代才放弃，
    `env.step()` 一下子变得很慢，主循环从 50Hz 掉到十几 Hz —— 手跟着一起卡。
    手的数据来自头显、指令走自己的 CAN 线，和臂没有任何共享资源，
    没有理由陪臂一起慢。拆开之后手固定按 `--hand-hz` 跑。
    """

    def __init__(self, streamer, retarget, hand, hz, freeze_thumb_rot, debug=False):
        super().__init__(name="atom-hand-track", daemon=True)
        self.streamer, self.retarget, self.hand = streamer, retarget, hand
        self.period = 1.0 / max(hz, 1.0)
        self.freeze_thumb_rot = freeze_thumb_rot
        self.debug = debug
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._angles = [1000] * NUM_DOF
        self._pinch = np.zeros(len(PINCH_FINGERS))
        self._raw = np.zeros(NUM_DOF)
        self._raw_lo = np.full(NUM_DOF, np.inf)
        self._raw_hi = np.full(NUM_DOF, -np.inf)
        self._ang_lo = [10**6] * NUM_DOF
        self._ang_hi = [-10**6] * NUM_DOF
        self._meas_lo = [10**6] * NUM_DOF
        self._meas_hi = [-10**6] * NUM_DOF
        self.ticks = 0
        self.hz_est = 0.0
        # 重定向这一段出异常时，症状和「手根本没接」一模一样：手一动不动。
        # 原来这里是 `except Exception: pass`，一个字都不吭 —— 查的时候
        # 没有任何线索可循。异常照样不许逃出线程（逃出去手会永久冻住），
        # 但必须数出来、把最后一条留下。
        self.errors = 0
        self.last_error = ""

    def snapshot(self):
        with self._lock:
            return list(self._angles), self._pinch.copy()

    def range_report(self):
        """整场跑下来每一维的实际量程。

        「在变」和「变得够不够」是两回事：raw 只在一小段里动的话，映射出来
        的角度变化很小，看上去就像没动。把 min/max 摆出来就不用猜了。
        """
        with self._lock:
            rlo, rhi = self._raw_lo.copy(), self._raw_hi.copy()
            alo, ahi = list(self._ang_lo), list(self._ang_hi)
            mlo, mhi = list(self._meas_lo), list(self._meas_hi)
        lines = [f"  {'维':<12}{'raw 范围':>16}{'指令角度':>14}{'实测角度':>14}   判定"]
        for i in range(NUM_DOF):
            span_a = ahi[i] - alo[i]
            span_m = mhi[i] - mlo[i]
            if span_a < 80:
                v = "✗ 指令就没怎么变（跟踪/标定问题）"
            elif span_m < 80:
                v = "✗ 指令在变但手没执行（机械/驱动问题）"
            else:
                v = "✓"
            lines.append(
                f"  {DOF_NAMES[i]:<12}{rlo[i]:6.2f}~{rhi[i]:<9.2f}"
                f"{alo[i]:5d}~{ahi[i]:<8d}{mlo[i]:5d}~{mhi[i]:<8d}   {v}")
        return "\n".join(lines)

    def debug_line(self):
        with self._lock:
            raw, ang, d = self._raw.copy(), list(self._angles), self._pinch.copy()
        curl = getattr(self.retarget, "last_curl", None)
        pin = getattr(self.retarget, "last_pinch", None)
        out = []
        for i in range(NUM_DOF):
            c = "" if curl is None else f"(弯{curl[i]})"
            p = "" if not pin or i not in pin else f"(捏{pin[i]})"
            out.append(f"{DOF_NAMES[i][:5]} raw={raw[i]:5.2f}->{ang[i]:4d}{c}{p}")
        out.append("到拇指cm " + " ".join(f"{v*100:.1f}" for v in d))
        return "  ".join(out)

    def stop(self):
        self._stop.set()

    def run(self):
        t_last = time.time()
        while not self._stop.is_set():
            t0 = time.time()
            data = self.streamer.get_latest()
            if data is not None:
                try:
                    ang, raw, d4 = self.retarget(np.asarray(data["left_fingers"]))
                    if self.freeze_thumb_rot and self.hand is not None:
                        lo, hi = self.hand.limits[THUMB_ROT]
                        ang[THUMB_ROT] = int((lo + hi) / 2)
                    meas = self.hand.measured() if self.hand is not None else None
                    with self._lock:
                        self._angles, self._raw, self._pinch = ang, raw, np.atleast_1d(d4)
                        self._raw_lo = np.minimum(self._raw_lo, raw)
                        self._raw_hi = np.maximum(self._raw_hi, raw)
                        for i in range(NUM_DOF):
                            self._ang_lo[i] = min(self._ang_lo[i], ang[i])
                            self._ang_hi[i] = max(self._ang_hi[i], ang[i])
                            if meas is not None:
                                self._meas_lo[i] = min(self._meas_lo[i], meas[i])
                                self._meas_hi[i] = max(self._meas_hi[i], meas[i])
                    if self.hand is not None:
                        self.hand.post(ang)
                    self.ticks += 1
                    if self.ticks % 20 == 0:
                        now = time.time()
                        self.hz_est = 20.0 / max(now - t_last, 1e-6)
                        t_last = now
                except Exception as exc:  # noqa: BLE001
                    # 绝不让异常杀掉线程（那会永久冻住整只手），但要留痕。
                    self.errors += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    if self.errors in (1, 10, 100) or self.errors % 1000 == 0:
                        print(f"\n[手] 重定向第 {self.errors} 次出错: "
                              f"{self.last_error}", flush=True)
            dt = time.time() - t0
            if dt < self.period:
                time.sleep(self.period - dt)


def recover_estop(env, args) -> bool:
    """清掉控制器锁存的急停状态，然后原地重试上电 + RT 启动。

    SDK 的复位接口：
        robot.recoverState(1, ec)    # 文档原文「恢复选项, 1 表示急停恢复」
        robot.clearServoAlarm(ec)    # 清伺服报警

    ⚠ 自动清急停是**放宽了一道安全机制**。急停存在的意义就是让机器停下，
      正常流程里该由人确认过原因再复位。这里做成开机自动，是因为这台臂
      没有物理急停按钮、只能走 API，而且复位 ≠ 会动：复位后还要上电，
      要动仍然得人挂离合 + 按住右手 dead-man。不想自动复位用 --no-estop-recover。

    返回 RT 是否最终起来了。
    """
    robot = getattr(env.arm, "_robot", None)
    if robot is None:
        return False
    print("\n[急停复位] recoverState(1) —— 控制器锁存的急停状态")
    ok = True
    for name, call in (("急停恢复", lambda ec: robot.recoverState(1, ec)),
                       ("清伺服报警", lambda ec: robot.clearServoAlarm(ec))):
        try:
            ec = {}
            call(ec)
            code = ec.get("ec", 0)
            if code:
                print(f"  {name}: ec={code} {ec.get('message', '')}")
            else:
                print(f"  {name}: 成功")
            if code:
                ok = False
        except Exception as e:
            print(f"  {name}: 调用失败 {type(e).__name__}: {e}")
            ok = False
    # 复位之后把上电和 RT 重新走一遍——驱动第一次是在急停状态下试的，必然失败
    try:
        env.arm._power_on()
    except Exception as e:
        print(f"  重新上电失败: {type(e).__name__}: {e}")
    try:
        env.arm._start_realtime()
    except Exception as e:
        print(f"  重启 RT 失败: {type(e).__name__}: {e}")
    active = bool(getattr(env.arm, "_rt_active", False))
    print(f"  结果: RT {'已启动' if active else '仍未启动'}")
    return active


def diagnose_stop(env, tag="停机诊断"):
    """臂突然不动/听到抱闸声时，把控制器自己的说法问出来。

    软件侧的 latch 和控制器侧的保护停机是两回事：latch 只是驱动拒绝发目标，
    控制器仍然带电；听到「咔」一声抱闸吸合、之后完全不受控，那是**控制器**
    自己停的，原因只能向它查。`queryEventInfo(Event.safety)` 里的 `collided`
    直接回答「是不是判了碰撞」——这是动力学模型不准时最常见的误触发。
    """
    print(f"\n── {tag} ──")
    robot = getattr(env.arm, "_robot", None)
    sdk = getattr(env.arm, "sdk", None)
    if robot is None or sdk is None:
        print("  没有 SDK 句柄，跳过")
        return
    for name, fn in (("电源", "powerState"), ("运行状态", "operationState")):
        try:
            ec = {}
            print(f"  {name}: {getattr(robot, fn)(ec)}")
        except Exception as e:
            print(f"  {name}: 查询失败 {type(e).__name__}: {e}")
    try:
        ec = {}
        info = robot.queryEventInfo(sdk.Event.safety, ec)
        collided = info.get(sdk.EventInfoKey.Safety.Collided)
        print(f"  安全事件: {info}")
        if collided:
            print("  ⚠ **控制器判定发生了碰撞**。臂没真撞上的话，多半是动力学模型不准：")
            print("     负载/质心填错，残余重力矩会被当成外部碰撞力。")
            print("     先用 --payload 0 跑一次对比——如果不再触发，就是负载值的问题。")
    except Exception as e:
        print(f"  安全事件: 查询失败 {type(e).__name__}: {e}")
    try:
        print(f"  RT 运动故障: {env.arm.has_motion_error()}")
    except Exception:
        pass
    try:
        q, m, k = joint_margins(env)
        print(f"  最紧关节 j{k} 余量 {np.rad2deg(m[k]):.1f}°")
    except Exception:
        pass


def apply_payload(env, args) -> None:
    """RT 起来之后再设负载，用**正确的**构造函数。

    不能交给 vel_ar5：它写的是 `self.sdk.Load()`，而 SDK 的 Load 只有
    `Load(mass, cog, inertia)` 一个构造函数，无参调用直接抛异常 ——
    而那个异常会被 `_start_realtime()` 的 except 吞成 warning，
    结果是整条 RT 起不来。所以上面把 `payload_mass` 留成 0 绕开它，
    在这里自己补。

    `setLoad` 的配置控制器会保存，但**机器人重启后恢复默认**，所以每次连接都要重设。
    """
    if args.no_tcp or args.mock or args.payload <= 0:
        return
    rt = getattr(env.arm, "_rt", None)
    if rt is None:
        print("  [负载] 没有 RT 控制器句柄，跳过")
        return
    try:
        load = env.arm.sdk.Load(float(args.payload),
                                [0.0, 0.0, float(args.payload_com_z)],
                                [0.0, 0.0, 0.0])
        ec = {}
        rt.setLoad(load, ec)
        if ec.get("ec", 0):
            print(f"  [负载] setLoad 返回 ec={ec.get('ec')} {ec.get('message','')}")
        else:
            print(f"  [负载] 已设 {args.payload:.2f} kg，质心 z={args.payload_com_z*1000:.0f} mm")
    except Exception as e:
        print(f"  [负载] 设置失败: {type(e).__name__}: {e}")
        print("        力控/碰撞检测会带着手的自重偏差，但运动本身不受影响。")


def require_realtime(env, args) -> bool:
    """连上之后立刻确认 RT 流真的起来了。起不来就当场停，别等到第一次接合才炸。

    vel_ar5 的 `_start_realtime()` 失败时**只打一条 warning** 就把
    `_rt_active` 置 False 继续跑。但它在失败点**之前**已经调过
    `setMotionControlMode(RtCommandMode)`，异常处理里没有还原 —— 于是：

        控制器：RT 模式        驱动自以为：排队模式

    `AR5Env.step()` 靠 `_rt_active` 选路，False 就走排队的 `moveAppend`，
    而排队指令在 RT 模式下会被直接拒绝：
        SDKError: moveAppend failed (ec=262): 运动控制模式错误,请切换到正确的模式

    症状出现在第一次接合（可能是几百步之后），跟真正的原因隔了很远。
    而且这个模式残留会**跨进程**：上一次崩掉的运行留下 RT 模式，
    下一次 startMove 更容易失败，越滚越糟。所以这里顺手把模式还原掉。
    """
    if getattr(env.arm, "_rt_active", False):
        return True
    if args.mock:
        return True                      # 假臂没有 RT，本来就走别的路

    # RT 没起来时，先看是不是急停锁着——是的话清掉再重试一次，别直接判死
    stuck_estop = bool(ERRLOG and ERRLOG.matching("急停"))
    if stuck_estop and not args.no_estop_recover:
        if recover_estop(env, args):
            return True

    print("\n" + "!" * 62)
    print("!! 实时(RT)控制没起来 —— 现在停下，不要继续。")
    print("!!")
    # 直接摆出驱动的原话，不要让人往回翻
    msgs = ERRLOG.matching("real-time control unavailable", "power on", "急停") if ERRLOG else []
    if msgs:
        print("!! 驱动报的原因：")
        for m in msgs[-3:]:
            print(f"!!   {m}")
    else:
        print("!! 往回翻一下 [RokaeAR5] real-time control unavailable (...) 那条 warning。")
    print("!!")
    blob = " ".join(msgs)
    hit = False
    for words, advice in FIX_HINTS:
        if any(w in blob for w in words):
            if "%s" in advice:
                advice = advice % getattr(env.config.arm, "local_ip", "?")
            print(f"!! → {advice}")
            hit = True
            break
    if not hit:
        print("!! 常见原因：急停被按下 / 臂没上电 / 上次崩溃把控制器留在 RT 模式 /")
        print("!!           local_ip 不对(报 1338 UDP socket打开失败) / 别的进程占着")
    print("!" * 62)

    # 把控制器还原成排队模式，免得这次的残留继续毒害下一次启动
    try:
        ec = {}
        env.arm._robot.setMotionControlMode(env.arm.sdk.MotionControlMode.NrtCommandMode, ec)
        print(f"\n  已尝试把控制器还原成 NrtCommandMode: ec={ec.get('ec')} "
              f"{ec.get('message', '')}")
        print("  重新运行本脚本再试一次。")
    except Exception as e:
        print(f"\n  还原控制模式失败: {type(e).__name__}: {e}")
        print("  可能需要在示教器上手动切回，或给臂重新上电。")
    return False


def soft_limits(env):
    """软限位带。优先用驱动算好的，没有就从 config 自己算（MockArm 没有那两个私有数组）。"""
    lo = getattr(env.arm, "_joint_lo", None)
    hi = getattr(env.arm, "_joint_hi", None)
    if lo is None or hi is None:
        a = env.config.arm
        margin = getattr(a, "joint_limit_margin", 0.0)
        lo = np.asarray(a.joint_min, dtype=float) + margin
        hi = np.asarray(a.joint_max, dtype=float) - margin
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def joint_margins(env):
    """每根关节离软限位还有多少弧度，以及最紧的那根。"""
    q = np.asarray(env.arm.get_joint_positions(), dtype=float)
    lo, hi = soft_limits(env)
    m = np.minimum(q - lo, hi - q)
    return q, m, int(np.argmin(m))


def print_posture(env, tag):
    """姿态体检。照搬 vel_ar5 teleop.py 的判据，不另发明。"""
    try:
        q, m, k = joint_margins(env)
        print(f"  [{tag}] 最紧关节 j{k} 余量 {np.rad2deg(m[k]):.1f}°   "
              + " ".join(f"j{i}:{np.rad2deg(v):.0f}°" for i, v in enumerate(m)))
    except Exception as e:
        print(f"  [{tag}] 读关节失败: {type(e).__name__}: {e}")
    try:
        r = env.posture_report()
        print(f"  [{tag}] 可操作度 {r['manipulability']:.4f}，"
              f"行程用得最多的是 j{r['worst_joint']} ({r['worst_usage']*100:.0f}%)")
        if r["near_singularity"]:
            print("        ⚠ 接近奇异，IK 会大面积解不出来")
        elif r["worst_usage"] > 0.85:
            print(f"        ⚠ j{r['worst_joint']} 几乎没有行程了")
    except Exception:
        pass


def connect_streamer(args):
    """连头显并等到第一帧。连不上返回 None。

    ⚠ 不能直接 `VisionProStreamer(ip=...)`：IP 不通时它**无限期阻塞**，
    什么都不打印，表现成「卡住/连不上」且无从查起。这里先用普通 socket
    2 秒内确认端口，再把构造丢线程里限时，失败要给得出**能照着做**的原因。
    所有输出都 flush —— 不 flush 的话重定向到文件/管道时块缓冲会把信息吞掉。
    """
    from probe_axes import connect_streamer_guarded
    return connect_streamer_guarded(args.ip, log=lambda m: print(m, flush=True))


def home_and_report(env, args):
    """启动时回起始姿态。

    起始姿态直接用 vel_ar5 的 `config.home_joints`（j5/j6 都摆在中位附近，
    余量 43°/39°），走 `env.home_joints()` —— **关节空间**移动，不需要 IK，
    所以从「IK 已经不收敛、腕关节顶在限位上」的状态也能出来。
    这正是笛卡尔指令救不回来的那种局面唯一的出路。
    """
    print_posture(env, "开机")
    if args.no_home:
        print("  --no-home：跳过回位。若上一轮卡在限位上，这次多半还是卡着。")
        return
    print(f"\n  ⚠ 即将回起始姿态，臂会移动。{args.home_wait:.0f} 秒内 Ctrl+C 可取消 ...",
          flush=True)
    try:
        time.sleep(args.home_wait)
    except KeyboardInterrupt:
        print("\n  已取消回位。")
        return
    print("  关节空间回位中 ...", end=" ", flush=True)
    try:
        ok = env.home_joints()
        print("完成" if ok is not False else "未在超时内到位")
    except Exception as e:
        print(f"失败: {type(e).__name__}: {e}")
        return
    if args.recentre:
        print("  移到工作空间中心 ...", end=" ", flush=True)
        try:
            print("完成" if env.move_to_workspace_centre() else "超时")
        except Exception as e:
            print(f"失败: {type(e).__name__}: {e}")
    try:
        env.reset(fake=True)          # 把指令位姿重新锚到实测，别留着旧目标
    except Exception:
        pass
    print_posture(env, "回位后")
    try:
        pinned = env.pinned_axes()
        if pinned:
            print("  ⚠ 工具贴着工作空间边界，这些方向的输入会被静默裁掉：")
            for axis, face, clear in pinned:
                print(f"      {axis} {face}: 余 {clear*1000:.0f} mm")
    except Exception:
        pass


# ────────────────────────────────────────────────────────────── 标定
def run_calibration(streamer, path, seconds=4.0):
    """录两个姿态的原始特征：完全张开 / 完全握拳。取中位数，抗抖。"""
    def grab(prompt):
        print(f"\n>>> {prompt}")
        for k in (3, 2, 1):
            print(f"    {k} ...", flush=True)
            time.sleep(1.0)
        print(f"    采集 {seconds:.0f} 秒，保持住不要动 ...", flush=True)
        buf, dbuf, t0 = [], [], time.time()
        while time.time() - t0 < seconds:
            d = streamer.get_latest()
            if d is not None:
                f = left_fingers(d)
                r = raw_features(f)
                if np.all(np.isfinite(r)):
                    buf.append(r)
                    dbuf.append(pinch_distances(f))
            time.sleep(0.02)
        if len(buf) < 10:
            raise RuntimeError(f"只采到 {len(buf)} 个有效样本，手没进视野？")
        arr = np.asarray(buf)
        med = np.median(arr, axis=0)
        spread = np.percentile(arr, 90, axis=0) - np.percentile(arr, 10, axis=0)
        dmed = np.median(np.asarray(dbuf), axis=0)          # 四根手指各自到拇指
        print("    " + "  ".join(f"{DOF_NAMES[i][:5]}={med[i]:.2f}±{spread[i]:.2f}"
                                 for i in range(NUM_DOF))
              + "   到拇指: " + " ".join(f"{DOF_NAMES[f][:4]}={dmed[k]*100:.1f}"
                                        for k, f in enumerate(PINCH_FINGERS)) + " cm")
        return med, spread, dmed

    print("=" * 62)
    print("手部标定 —— 左手全程保持在头显视野里")
    print("=" * 62)
    op, sp_o, d_open = grab("把左手**完全张开**，五指伸直、拇指自然外展")
    cl, sp_c, _ = grab("把左手**握成拳**，拇指压在食指上（对掌）")
    # 第三个姿态专门给捏合：握拳时弯曲量够用，但捏取时弯曲量看着才「半闭」，
    # 必须另外拿指尖距离来驱动，否则机械手永远够不到接触点。
    _, _, d_pin = grab("**拇指和食指指尖捏在一起**，其余手指自然伸着")
    # 指尖相触的距离对哪根手指都差不多，所以只用食指这一个姿态定「相触」端点，
    # 四根共用；「张开」端点则是张开姿态下各自实测的（小指本来就离拇指最远）。
    d_pinch = float(d_pin[0])

    bad = [i for i in range(NUM_DOF) if abs(cl[i] - op[i]) < 0.15]
    if bad:
        print("\n⚠ 这几维两个姿态几乎没区别，映射会失效：",
              ", ".join(DOF_NAMES[i] for i in bad))
        print("  多半是手没完全张开/握紧，或者手指被遮挡。建议重做。")
    noisy = [i for i in range(NUM_DOF)
             if max(sp_o[i], sp_c[i]) > 0.4 * max(abs(cl[i] - op[i]), 1e-6)]
    if noisy:
        print("⚠ 这几维抖动相对量程偏大：", ", ".join(DOF_NAMES[i] for i in noisy))

    thin = [DOF_NAMES[f] for k, f in enumerate(PINCH_FINGERS) if d_open[k] - d_pinch < 0.02]
    if thin:
        print(f"\n⚠ 这几根手指的「张开-相触」距离差太小，捏合驱动会失效: {thin}")

    c = Calibration(raw_open=list(op), raw_closed=list(cl),
                    pinch_open=list(d_open), pinch_closed=d_pinch)
    c.save(path)
    print(f"\n已写入 {path}")
    print("指尖距离端点(cm): 相触 %.1f  张开 " % (d_pinch*100)
          + " ".join(f"{DOF_NAMES[f][:4]}={d_open[k]*100:.1f}"
                     for k, f in enumerate(PINCH_FINGERS)))
    print("映射自检（原始值 -> 机械手角度）：")
    print(f"  张开 -> {c.to_angles(op)}")
    print(f"  握拳 -> {c.to_angles(cl)}")
    pa = c.pinch_angles([d_pinch] * len(PINCH_FINGERS))
    print("  四指各自与拇指捏合时 -> "
          + "  ".join(f"{DOF_NAMES[d]}={v}" for d, v in sorted(pa.items()))
          + "   （接触点附近，靠力控停住）")
    return c


# ────────────────────────────────────────────────────────────── 主流程
def main():
    args = parse_args()
    # 驱动里那条「real-time control unavailable」是 warning，不配日志很容易漏掉
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    # 臂的 IK/限位 warning 是每帧一条、50Hz 刷，会把状态行冲掉
    for h in logging.getLogger().handlers:
        h.addFilter(RateLimit(args.log_every))
    global ERRLOG
    ERRLOG = LastErrors()
    logging.getLogger().addHandler(ERRLOG)   # 限流之外单独留一份原始报错

    # --calib 只需要头显，不碰臂，单独走一条路
    if args.calib:
        streamer = connect_streamer(args)
        if streamer is None:
            return 1
        run_calibration(streamer, args.calib_file)
        return 0

    calib_path = Path(args.calib_file)
    if calib_path.exists():
        calib = Calibration.load(calib_path)
        print(f"已载入标定 {calib_path}")
        # 标定文件里存了 limits 的快照，会盖掉代码里的默认值。
        # 改过 DEFAULT_LIMITS 却忘了重标，就会静默按旧钳位跑。
        from inspire_hand6 import DEFAULT_LIMITS as _DL
        from inspire_hand6 import INDEX as _I2, THUMB_BEND as _T2
        drift = [i for i in range(NUM_DOF)
                 if i not in (_I2, _T2)
                 and tuple(calib.limits.get(i, ())) != tuple(_DL[i])]
        if drift:
            print("⚠ 标定文件里的安全钳位和当前代码不一致（文件里的生效）：")
            for i in drift:
                print(f"    {DOF_NAMES[i]:<11} 文件 {tuple(calib.limits[i])}  代码 {tuple(_DL[i])}")
            print("    重跑一次 --calib 就会写入新值。")
        if "pinch_open" not in calib_path.read_text(encoding="utf-8"):
            print("⚠ 这份标定没有捏合端点（是加捏合驱动之前标的），正在用通用兜底 9cm/1.5cm。")
            print("    重跑 --calib 会多采一个「拇指食指捏在一起」的姿态，捏取精度全靠它。")
    else:
        calib = Calibration()
        print(f"⚠ 没有标定文件 {calib_path}，用的是通用兜底端点 —— 先跑一次 --calib")
    retarget = Retargeter(calib, smooth=args.smooth, pinch_gate=args.pinch_gate)

    # ── 臂 ──
    config = AR5Config(mock=args.mock, real_hand_in_mock=False)
    config.use_hand = False                    # 手由本项目自己的驱动接管
    config.arm.use_realtime = True
    config.max_lead = float(args.max_lead)          # 缰绳放宽，遥操才跟得上
    if args.arm_speed > 0:
        config.arm.max_joint_speed = float(args.arm_speed)
    if not args.no_tcp:
        # 官方 RH56E2：带触觉 790g±10g，掌部 93.0 x 211.5mm。
        # tcp_offset 是「末端执行器坐标系相对法兰」的位姿(米+轴角)，
        # 这里只给沿法兰法向的平移，姿态不转。这一段 vel_ar5 处理得对。
        config.arm.tcp_offset = (0.0, 0.0, float(args.tcp_z), 0.0, 0.0, 0.0)
        # ⚠ payload_mass 必须留 0，负载由本脚本在 RT 起来之后自己设。
        #
        # vel_ar5 的 _apply_rt_safety() 里那段：
        #     if self.config.payload_mass > 0:
        #         load = self.sdk.Load()          # ← SDK 没有无参构造函数
        # 只在 payload_mass 非零时才执行，所以一直没人踩到。一旦填了非零值，
        # `Load()` 立刻抛 incompatible constructor arguments，异常被
        # _start_realtime() 的 except 吞成一条 warning，RT 就起不来了
        # （然后第一次接合时以 ec=262 的形式爆出来，离真正原因隔了几百步）。
        config.arm.payload_mass = 0.0
        config.arm.payload_com = (0.0, 0.0, float(args.payload_com_z))
    if args.no_cameras:
        config.cameras.sources = {}
    if args.mock:
        config.settle_time = 0.0
    env = AR5Env(config)
    env.connect()
    if not require_realtime(env, args):
        env.close()
        return 2
    # 换掉 IK。驱动里只有一个调用点（rokae_arm.py:847），
    # 换掉 _kinematics 就够了，不用碰 vel_ar5 的文件。
    # ── 把整条位姿链从法兰系搬到工具系 ──────────────────────────────────
    # `--tcp-z` 原来只通过 setEndEffectorFrame 告诉了**控制器**（给力控/碰撞用），
    # 本地运动学一无所知：fk/ik 和 get_ee_pose 全在法兰系。
    # 后果不是一个常数偏移，是**旋转中心错了** —— 手腕原地转，法兰不动，
    # 而灵巧手在 150mm 外画弧（实测 30° → 69mm）。操作者感受就是「反解不对」。
    # 换系必须**成对**做：kinematics 和 get_ee_pose 一起，少一个就是新的不一致。
    tool_z = 0.0 if args.no_tcp else float(args.tcp_z)
    if tool_z > 0 and not args.flange_frame and not args.mock:
        _Xt = np.eye(4)
        _Xt[:3, 3] = np.array([0.0, 0.0, tool_z])
        _raw_pose = env.arm.get_ee_pose

        def _tool_pose(_raw=_raw_pose, _X=_Xt):
            p6 = np.asarray(_raw(), dtype=float)
            T = np.eye(4)
            T[:3, :3] = axis_angle_to_matrix(p6[3:6])
            T[:3, 3] = p6[:3]
            T = T @ _X
            out = np.empty(6)
            out[:3] = T[:3, 3]
            out[3:6] = matrix_to_axis_angle(T[:3, :3])
            return out

        env.arm.get_ee_pose = _tool_pose
        if getattr(env.arm, "_kinematics", None) is not None:
            env.arm._kinematics = ToolFrameKinematics(
                env.arm._kinematics, (0.0, 0.0, tool_z, 0.0, 0.0, 0.0))
        print(f"  位姿链: **工具系**（法兰外 {tool_z*1000:.0f}mm）—— 手腕原地转，手就原地转")
        print("         ⚠ 工作空间盒 workspace_min/max 原来是按法兰定的，"
              "现在约束的是工具，等于整体前移了")
    elif tool_z > 0:
        print("  位姿链: 法兰系（--flange-frame）—— 手腕原地转会把手甩出去")

    mode = "stock" if args.stock_ik else args.ik
    if mode != "stock" and getattr(env.arm, "_kinematics", None) is not None:
        margin = getattr(config.arm, "joint_limit_margin", 0.05)
        lock = {5: float(args.lock_j5)} if args.lock_j5 is not None else None
        installed = None
        if lock and mode == "srs":
            # 闭式解的 j5 由腕部 ZYZ 分解唯一确定，没法自由固定 ——
            # 固定了就等于少一个姿态自由度，闭式解直接无解。
            # 实测：--ik srs --lock-j5 0 时 j5 照样走了 21.9°（锁只在退路的优化式里）。
            print("  IK: --lock-j5 和 --ik srs 不能共存（闭式解的 j5 由腕部分解定死），"
                  "本次改用优化式")
            mode = "opt"
        if mode == "poe":
            from pathlib import Path as _P
            if not (_P(__file__).resolve().parent / "ar5_dh.json").exists():
                print("  IK: 缺 ar5_dh.json，先跑 `python read_dh.py`。本次退回优化式。")
                mode = "opt"
            else:
                from vel_kin import AR5Kinematics as _K
                import json as _json
                _d = _json.loads((_P(__file__).resolve().parent
                                  / "ar5_dh.json").read_text(encoding="utf-8"))
                jn, jx = config.arm.joint_min, config.arm.joint_max
                _cal = _K(_d["calibrated"], 7, jn, jx)
                jump_max = getattr(config.arm, "joint_jump_max", 0.15)
                step = min(_auto_step(config, args), 0.9 * jump_max)
                q_rest = np.asarray(config.arm.home_joints, dtype=float)
                env.arm._kinematics = AR5PoeIK(
                    calibrated=_cal, limit_margin=margin, max_step=step, lock=lock,
                    fallback=AR5OptIK(_cal, limit_margin=margin, max_step=step,
                                      q_rest=q_rest, lock=lock))
                installed = "poe"
                print("  IK: 旋量法解析解（PoE + Paden-Kahan；够不到时退数值解打滑）")
                if lock:
                    print(f"      j5 目标 {np.rad2deg(args.lock_j5):+.1f}° —— "
                          "闭式解靠**筛分支**逼近，筛不出来时由退路的数值解硬锁")
        if installed is None and mode == "srs":
            import json as _json
            from pathlib import Path as _P
            from vel_kin import AR5Kinematics as _K
            dhf = _P(__file__).resolve().parent / "ar5_dh.json"
            if not dhf.exists():
                print(f"  IK: 缺 {dhf.name}，先跑 `python read_dh.py`。本次退回优化式。")
                mode = "opt"
            else:
                _d = _json.loads(dhf.read_text(encoding="utf-8"))
                jn, jx = config.arm.joint_min, config.arm.joint_max
                _nom = _K(_d["nominal"], 7, jn, jx)
                _cal = _K(_d["calibrated"], 7, jn, jx)
                jump_max = getattr(config.arm, "joint_jump_max", 0.15)
                step = min(_auto_step(config, args), 0.9 * jump_max)
                q_rest = np.asarray(config.arm.home_joints, dtype=float)
                env.arm._kinematics = AR5SrsIK(
                    _nom, _cal, limit_margin=margin, max_step=step,
                    fallback=AR5OptIK(_cal, limit_margin=margin, max_step=step,
                                      q_rest=q_rest, lock=lock))
                installed = "srs"
                print("  IK: 闭式 S-R-S（分支枚举 + 离上一帧最近；够不到时退优化式打滑）")
                print("      ⚠ 实验性：精度优于优化式(0.000mm)，但跳变更大(3.8° vs 1.2°)")
        if installed is None and mode == "opt":
            # 步长上限必须真的小于驱动那道闸，顶着设会让舍入决定成败
            # （config.py 里 joint_goal_step 的注释记过这个教训：
            #  1.926388-0.15 回来是 0.15000000000000013，闸门判成跳变，
            #  当时 100% 的目标被拒、臂一动不动）。
            jump_max = getattr(config.arm, "joint_jump_max", 0.15)
            step = min(_auto_step(config, args), 0.9 * jump_max)
            # 冗余消解拉向 **home 姿态**，不是拉向 joint_mid。
            # 7 轴解 6 维位姿，零空间有 1 维（肘部绕「肩—腕」连线转）。
            # 位姿定不住肘部，必须另外给个偏好，否则肘会在零空间里随机游走：
            # TCP 明明没怎么动，肘却自己换边——操作者完全无法预期。
            # joint_mid 是纯几何中点，对应的臂型不一定是人能理解的；
            # home_joints 是挑过的工作姿态（TCP 在工作空间盒正中，
            # 没有一个轴用超过 68% 行程），拿它当吸引子，肘就稳定停在一个
            # 人看得懂的位置上。
            q_rest = np.asarray(config.arm.home_joints, dtype=float)
            env.arm._kinematics = AR5OptIK(env.arm._kinematics,
                                           limit_margin=margin, max_step=step,
                                           q_rest=q_rest, lock=lock)
            installed = "opt"
            if lock:
                print(f"      j5 锁死在 {np.rad2deg(args.lock_j5):+.1f}° "
                      f"—— 只用其余 6 轴解，冗余消失")
            print(f"  IK: 优化式（位姿误差当代价，限位±{margin} 和步长{step:.3f}rad 当硬约束；"
                  f"够不到就打滑，不丢帧）")
        if installed is None:
            # ⚠ 这里以前是 `else:`，挂在 `if mode == "opt"` 上。
            # mode=="srs" 时 opt 分支为假 → 掉进 else → **刚装好的 SrsIK 被 DLS 覆盖**，
            # 而横幅还照常打印 srs。锁 j5 的代码也只在 opt 分支里，于是一并失效。
            # 改成 installed 哨兵，三个分支互斥，装了就不再被谁覆盖。
            env.arm._kinematics = AR5IK(env.arm._kinematics, limit_margin=margin)
            installed = "dls"
            print(f"  IK: 雅可比迭代改进版（自适应阻尼 + WLN 限位规避 + 误差钳位 + 早退，"
                  f"按软限位±{margin}求解）")

    # 装完之后**实测**一次：直接问求解器要一个解，看它到底是谁、j5 锁没锁。
    # 之前没有这一步，所以「装了 SrsIK 又被 DLS 覆盖」这种事在日志上完全看不出来
    # （横幅照常打印 srs），只能等到真机上表现不对才发现。
    try:
        _k = getattr(env.arm, "_kinematics", None)
        if _k is not None and not args.mock:
            _q0 = np.asarray(env.arm.get_joint_positions(), dtype=float)
            _T = _k.fk(_q0).copy()
            _T[:3, 3] += np.array([0.01, 0.01, 0.0])      # 只是问一下，不下发
            _q1, _ = _k.ik(_T, _q0)
            print(f"  [自检] 实际求解器 = {type(_k).__name__}")
            if _q1 is not None:
                _dj5 = abs(float(_q1[5] - _q0[5]))
                if args.lock_j5 is not None:
                    _off = abs(float(_q1[5] - args.lock_j5))
                    print(f"  [自检] j5 锁定验证: 解出 j5={np.rad2deg(_q1[5]):+.3f}°，"
                          f"距设定值 {np.rad2deg(_off):.4f}°  "
                          f"{'✓ 锁住了' if _off < 1e-6 else '✗ **没锁住**'}")
                else:
                    print(f"  [自检] j5 未锁（没给 --lock-j5）；这一小步 j5 动了 "
                          f"{np.rad2deg(_dj5):.3f}°")
            else:
                print("  [自检] 求解器对一个 1cm 的小步都给不出解 —— 不正常，先查这个")
    except Exception as _e:
        print(f"  [自检] 跳过: {type(_e).__name__}: {_e}")

    # ── MoveL 流式后端（只构造，配置推迟到回位之后）─────────────────────
    # ⚠ 顺序要命：`configure()` 会把控制器切到 NrtCommandMode，而此刻
    # require_realtime() 起的 **RT 发送线程还在 1kHz 压关节指令**。
    # 两边同时上会互相打架 —— `_stop_rt_thread()` 的注释原话：
    # "two threads pushing into one RT session ... alternate setpoints from
    #  independent trackers"。表现就是**臂完全不动**。
    # 所以这里只建对象；停 RT + 切模式 + 配置，全部放到 home_and_report 之后。
    movel = None
    if args.arm_backend == "movel":
        if args.mock:
            print("  [MoveL] --mock 下没有真实控制器，退回 rt 后端")
            args.arm_backend = "rt"
        else:
            from movel_stream import MoveLStreamer
            _raw = getattr(env.arm, "_robot", None)
            _sdk = getattr(env.arm, "sdk", None)
            if _raw is None or _sdk is None:
                print("  [MoveL] 拿不到 SDK 句柄，退回 rt 后端")
                args.arm_backend = "rt"
            else:
                movel = MoveLStreamer(_raw, _sdk, speed=args.movel_speed,
                                      zone=args.movel_zone,
                                      lead_ms=args.movel_lead_ms,
                                      min_step_mm=args.movel_deadband)

    apply_payload(env, args)
    home_and_report(env, args)          # ← 回位提前到这里：不依赖头显

    if movel is not None:
        # 回位已经走完（RT 流做的），现在把 RT 发送线程**彻底停掉**再交给 MoveL。
        # 不停的话它会 1kHz 压着「保持当前关节角」，和 MoveL 抢同一台臂。
        print("\n  [MoveL] 停掉 RT 关节流，切到排队式 ...")
        # 拆 RT 是**三步**，缺一不可。驱动自己的 disconnect() 就是这么写的
        # （rokae_arm.py:644-649）。只停发送线程的话，控制器还留在 RT 控制状态，
        # 后面 setMotionControlMode / setPowerState 全部被拒 `ec=-20 机器人运动中`，
        # 而且是**静默失败**。实机上就卡在这里。
        try:
            env.arm._stop_rt_thread()                    # ① 停发送线程
            env.arm._rt_active = False
            print("  [MoveL] ✓ ① 发送线程已停")
            _rtc = getattr(env.arm, "_rt", None)
            if _rtc is not None:
                _rtc.stopMove()                          # ② 停 RT 运动
                print("  [MoveL] ✓ ② RT 运动已停")
            _ec = {}
            env.arm._robot.setMotionControlMode(
                env.arm.sdk.MotionControlMode.NrtCommandMode, _ec)   # ③ 切排队模式
            if _ec.get("ec", 0):
                raise RuntimeError(f"ec={_ec.get('ec')} {_ec.get('message','')}")
            print("  [MoveL] ✓ ③ 已切到 NrtCommandMode")
        except Exception as exc:                        # noqa: BLE001
            print(f"  [MoveL] ✗ 拆 RT 失败: {type(exc).__name__}: {exc}")
            print("        两套控制会打架，臂不会动。改用 --arm-backend rt")
            movel = None
        if movel is not None:
            if not movel.configure():
                print("  [MoveL] 配置没过，不能用这个后端。改用 --arm-backend rt")
                movel = None
            else:
                movel.start()
            print(f"  臂后端: **MoveL 多路点流式**（逆解在控制器）"
                  f"  速度 {args.movel_speed:.0f}mm/s  延迟预算 {args.movel_lead_ms:.0f}ms"
                  f"  抖动死区 {args.movel_deadband:.1f}mm")

    # 臂就位了再连头显。放在后面是因为回位跟头显毫无关系，
    # 而头显那头（戴上/点 Start/IP 变了）经常要折腾几次——
    # 没必要让臂一直停在上一次的姿态等人。
    streamer = connect_streamer(args)
    if streamer is None:
        env.close()
        return 1

    # ── 手 ──
    hand = None
    if args.mock_hand:
        from inspire_hand6 import MockSerial
        hand = InspireHand6(port="(mock)", serial_factory=MockSerial)
        hand.connect()
    elif not args.no_hand and (not args.mock or args.real_hand):
        hand = InspireHand6(port=args.hand_port)
        hand.connect()
    if hand is not None:
        # 钳位有两个来源（驱动默认值、标定文件快照）。取**交集**：
        # 陈旧的标定文件只能把安全带收紧，绝不可能放宽。
        #
        # 但食指/拇指弯曲**除外**：这两维现在走耦合下限（下限随拇指位置变），
        # 老标定文件里存的静态下限(520/320 之类)会把耦合整个作废，
        # 表现就是「拇指张着单独弯食指也只能走一半」。
        from inspire_hand6 import INDEX as _IDX, THUMB_BEND as _TB
        for i in range(NUM_DOF):
            if i in (_IDX, _TB):
                calib.limits[i] = (0, 1000)      # 交给驱动的耦合规则
                hand.limits[i] = (0, 1000)
                continue
            clo, chi = calib.limits.get(i, hand.limits[i])
            hlo, hhi = hand.limits[i]
            hand.limits[i] = (max(clo, hlo), min(chi, hhi))
        ok, why = hand.liveness()
        if not ok:
            print("\n" + "!" * 62)
            print("!! 灵巧手的控制循环没在跑 —— 现在跟踪也没用，手指不会动。")
            print(f"!!   {why}")
            print("!! 症状：写寄存器一切正常（能原样读回），但 ANGLE_ACT 冻在一个")
            print("!!       常数上，指令怎么变手指都不动。")
            print("!! → **给手断电重启**（手是独立 24V 供电，不是 USB 供的）。")
            print("!!   重启后重跑本脚本；仍然这样就查 24V 供电和手内部。")
            print("!" * 62 + "\n")
        hand.set_force(args.hand_force)
        hand.set_speed(args.hand_speed)
        hand.start_writer()
        print(f"灵巧手已连接 {args.hand_port}  力控 {args.hand_force}g  速度 {args.hand_speed}")
        print("  安全带 " + "  ".join(f"{DOF_NAMES[i]}:{hand.limits[i][0]}-{hand.limits[i][1]}"
                                      for i in range(NUM_DOF)))

    tracker = HandTracker(streamer, retarget, hand, args.hand_hz,
                          args.freeze_thumb_rot, debug=args.hand_debug)
    tracker.start()

    R = yaw_matrix(args.yaw) @ mirror_matrix(args.mirror)
    jit = {"raw": [], "filt": [], "qcmd": [], "qact": []} if args.jitter_probe else None
    if args.arm_smooth > 0:
        pos_filt = OneEuro(args.arm_smooth, args.arm_smooth_beta)
        rot_filt = OneEuro(args.arm_smooth, args.arm_smooth_beta)
    else:
        pos_filt = rot_filt = None
    armed = bool(args.auto_arm)
    deadman = False
    ref = None
    movel_started = False
    movel_res = ""
    p_target = None
    R_target = None
    prev_frame = None
    last_fresh = time.time()
    stale = False
    steps = engaged_steps = stale_steps = rot_held_n = 0
    rot_dropped_sum = 0.0
    limit_rej_seen = limit_rej_run = stuck_recoveries = 0
    settle_frames = 0
    latch_warned = False
    power_warned = False
    unreach_seen = slip_run = slips = slip_burst = 0
    slip_t0 = slip_msg_t = 0.0
    slip_stuck_warned = False
    tight_note = ""
    cmd_travel = 0.0
    last_hand_send = 0.0
    hand_period = 1.0 / max(args.hand_hz, 1.0)

    # 开机横幅：把「什么是真的」摆到台面上。静默地没有手是最难查的故障，
    # 之前就因为 --mock 不带 --real-hand 白跑过一轮。
    arm_kind = "假臂(mock)" if args.mock else "真臂 " + config.arm.ip
    if hand is None:
        hand_kind = "【无】—— 手不会动"
        if not args.no_hand:
            hand_kind += "  ← 想接真手要加 --real-hand"
    else:
        hand_kind = "假手(mock 串口)" if args.mock_hand else "真手 " + args.hand_port
    print()
    print("┌" + "─" * 58)
    print(f"│ 臂 : {arm_kind}")
    print(f"│ 手 : {hand_kind}")
    tcpz = config.arm.tcp_offset[2]
    pm = 0.0 if args.no_tcp else float(args.payload)
    print("│ TCP : " + ("【未设】绕法兰转，手在外面画大弧" if args.no_tcp
                         else f"z={tcpz*1000:.0f}mm"))
    print("│ 负载: " + ("【未设】—— 填错会被控制器当成碰撞而保护停机，测准了再填"
                       if pm <= 0 else
                       f"{pm:.2f}kg  质心z={config.arm.payload_com[2]*1000:.0f}mm"))
    print(f"│ 标定: {'已载入 ' + str(calib_path) if calib_path.exists() else '【无】用兜底端点'}")
    print("└" + "─" * 58)
    print()
    print("  ENTER  挂/摘离合     h  关节空间回起始姿态（卡住时按它）     q  退出")
    print("  臂要动：【离合已挂(按 ENTER)】且【右手保持捏合】。手始终跟随左手五指。")
    print(f"  {args.hz} Hz  缩放 x{args.scale}  yaw {args.yaw}°  镜像 {args.mirror}  "
          f"姿态跟随 {'开' if args.with_rotation else '关'}")
    print(f"  地板 z >= {config.arm.joint_floor_z:.3f} m（只算到法兰+90mm，不含手）")
    print()

    interactive = sys.stdin.isatty()
    old_term = termios.tcgetattr(sys.stdin) if interactive else None
    if not interactive:
        print("  [提示] stdin 不是终端，键盘离合不可用")
    period = 1.0 / args.hz
    print_every = int(args.hz / args.print_hz) if args.print_hz > 0 else 0
    t_start = time.time()
    loop_hz = 0.0
    _hz_t0 = time.time()

    try:
        if interactive:
            import tty
            tty.setcbreak(sys.stdin.fileno())
        while True:
            t0 = time.time()
            if args.seconds and t0 - t_start >= args.seconds:
                break

            if interactive and select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "q":
                    break
                if ch in ("h", "H"):
                    # 关节空间回位：不需要 IK，所以从「IK 已经不收敛、腕关节顶死」
                    # 的状态也能出来。这是那种局面唯一的自救手段，做成随时可按，
                    # 免得每次都要退出重启。
                    if armed:
                        armed = False
                        print("\n[回位] 先摘离合")
                    ref = None
                    print("[回位] 关节空间回起始姿态 ...", end=" ", flush=True)
                    try:
                        if getattr(env.arm, "latched", False):   # MockArm 没这个属性
                            env.arm.clear_latch()
                        env.home_joints()
                        env.reset(fake=True)
                        print("完成")
                        print_posture(env, "回位后")
                    except Exception as e:
                        print(f"失败: {type(e).__name__}: {e}")
                    limit_rej_run = 0
                    continue
                if ch in ("\r", "\n"):
                    armed = not armed
                    ref = None
                    note = ""
                    if armed:
                        # latch（超限/超时后臂自锁）只能由操作者的明确动作解除，
                        # 挂离合正是这样一个动作。否则上一轮撞限位后要重启进程才能再动。
                        try:
                            if env.arm.latched:
                                env.arm.clear_latch()
                                note = "  (已解除上一次的 latch)"
                        except Exception as e:
                            note = f"  (解 latch 失败: {type(e).__name__})"
                    print(f"\n[离合] {'挂上' if armed else '摘下 —— 臂已停'}{note}")

            frame = streamer.latest
            if frame is not None and frame is not prev_frame:
                prev_frame = frame
                last_fresh = t0
                if stale:
                    print("\n[链路] 恢复")
                stale = False
            elif t0 - last_fresh > args.max_stale:
                if not stale:
                    print(f"\n[链路] 超过 {args.max_stale}s 没有新帧 —— 臂保持、手保持")
                stale = True

            data = streamer.get_latest()
            if data is None:
                time.sleep(period)
                continue

            # 手由独立线程跟踪（见 HandTracker），这里只取它的最新状态来显示。
            angles, pinch_d = tracker.snapshot()

            # ── 臂：右手捏合当 dead-man ──
            r_pinch = float(data["right_pinch_distance"])
            deadman = r_pinch < (PINCH_OFF if deadman else PINCH_ON)

            here = wrist_xyz(data, "left")
            Rw = wrist_R(data, "left") if args.with_rotation else None
            # 去抖。原来这里是原样进 IK 的，头显噪声直接变成关节指令。
            raw_here = here.copy()
            if pos_filt is not None:
                here = pos_filt(here, t0)
                if Rw is not None:
                    Rw = filter_rotation(rot_filt, Rw, t0)
            if jit is not None:
                jit["raw"].append(raw_here)
                jit["filt"].append(here.copy())
                try:
                    jit["qcmd"].append(np.asarray(env.arm._q_cmd, dtype=float).copy()
                                       if getattr(env.arm, "_q_cmd", None) is not None
                                       else np.full(7, np.nan))
                    jit["qact"].append(np.asarray(env.arm.get_joint_positions(),
                                                  dtype=float).copy())
                except Exception:
                    pass
            engaged = armed and deadman and not stale

            # ── 臂的跟踪：绝对位姿伺服 ────────────────────────────────────
            # 不再用「本帧相对上帧」的增量累加。增量一旦被吃掉就永远找不回来：
            # max_step_rotation 每步只放 0.1rad、缰绳清零、被限位拒绝的那步作废、
            # 自救重锚之后对应关系彻底断 —— 结果就是手和末端越偏越多，转不动。
            #
            # 改成：接合瞬间记下「手的位姿」和「末端的位姿」作为基准，之后每帧算出
            # **绝对目标位姿**，再把「目标 - 当前指令」作为这一帧的 delta 下发。
            # step() 再怎么钳位也只是让它慢一点到位，剩余误差下一帧继续发，自己会追回来。
            delta = np.zeros(6)
            rot_held = ""
            dropped = 0.0
            rot_cmp = None
            if engaged and ref is None:
                # 接合的上升沿：抓基准。用**指令位姿**而不是实测，避免把跟踪滞后
                # 一次性算进基准里（臂还没走到时实测是落后的）。
                base6 = np.asarray(getattr(env, "_target_pose", None)
                                   if getattr(env, "_target_pose", None) is not None
                                   else env.arm.get_ee_pose(), dtype=float)
                ref = {"p_hand": here.copy(),
                       "R_hand": None if Rw is None else Rw.copy(),
                       "p_tool": base6[:3].copy(),
                       "R_tool": axis_angle_to_matrix(base6[3:6])}
                print(f"\n[接合] 已锁定基准，从现在起「手什么姿态，末端就什么姿态」")

            if engaged and ref is not None:
                cur6 = np.asarray(getattr(env, "_target_pose", None)
                                  if getattr(env, "_target_pose", None) is not None
                                  else env.arm.get_ee_pose(), dtype=float)

                # 位置：手相对基准移了多少，末端就相对基准移多少
                #
                # --delta-frame tool 时，再额外乘上「末端自接合以来转了多少」，
                # 于是映射跟着末端一起转。接合那一刻 R_follow=I，两种模式完全一致，
                # 差别只在末端转起来之后。
                R_cur_for_map = axis_angle_to_matrix(cur6[3:6])
                if args.delta_frame == "tool":
                    R_follow = R_cur_for_map @ ref["R_tool"].T
                else:
                    R_follow = np.eye(3)
                p_target = ref["p_tool"] + R_follow @ R @ (here - ref["p_hand"]) * args.scale
                delta[:3] = (p_target - cur6[:3]) * args.gain

                if args.with_rotation and Rw is not None and ref["R_hand"] is not None:
                    # 姿态：手相对基准转了多少，末端就相对基准转多少（换到基座系）
                    R_rel = R @ (Rw @ ref["R_hand"].T) @ R.T
                    if args.delta_frame == "tool":
                        # 姿态也跟着末端系走：左乘(绕基座轴) → 共轭到末端当前朝向上
                        R_rel = R_follow @ R_rel @ R_follow.T
                    if args.rot_scale != 1.0:      # 缩放：轴角按比例缩
                        R_rel = axis_angle_to_matrix(matrix_to_axis_angle(R_rel) * args.rot_scale)
                    R_target = R_rel @ ref["R_tool"]

                    R_cur = axis_angle_to_matrix(cur6[3:6])
                    # 对照：手相对基准转了多少 vs 末端相对基准转了多少。
                    # 映射正确的话这两组数应该一致（末端滞后一点是正常的）。
                    # 哪个轴对不上，就是那个轴的映射有问题——不用猜。
                    rot_cmp = (np.rad2deg(matrix_to_axis_angle(R @ (Rw @ ref["R_hand"].T) @ R.T)),
                               np.rad2deg(matrix_to_axis_angle(R_cur @ ref["R_tool"].T)))
                    err = matrix_to_axis_angle(R_target @ R_cur.T) * args.gain  # 还差多少
                    raw_rot = float(np.linalg.norm(err))
                    if args.rot_mode == "roll":
                        # 只让末端跟手掌的「拧」，上下挥手不跟。
                        z = R_cur[:, 2]
                        err = z * float(err @ z)
                    dropped = raw_rot - float(np.linalg.norm(err))

                    # 姿态缰绳：臂侧的 _leash/_base_pose 重锚只看平移，姿态没人管。
                    # 绝对伺服下目标本身有界（跟着人的手），不会无界跑飞，
                    # 但臂够不到时误差会一直挂着，所以仍然限一下单帧下发量。
                    if raw_rot > args.max_rot_lead:
                        err = err / max(raw_rot, 1e-9) * args.max_rot_lead
                        rot_held = f" [姿态限幅 {np.rad2deg(raw_rot):.0f}°]"
                        rot_held_n += 1
                    delta[3:6] = err

            if engaged:
                engaged_steps += 1
            else:
                ref = None                     # 松手即弃基准，重新捏合时就地重锚

            # ── 打滑：臂跟不上时让基准跟着臂走 ────────────────────────────
            # 绝对伺服有个致命面：目标锚在人手上，人手一直动，误差就一直涨，
            # 把**臂根本到不了的目标**反复砸过去。实测日志里 Δ 从 10mm 一路涨到
            # 35mm，末端却一动不动，IK 每帧都 did not converge——而 IK 失败时
            # `servo_to_pose_rt` 直接返回 False、**`_q_goal` 根本没更新**，
            # RT 发送器于是判定「目标断流」，臂又正在动 → latch。
            #
            # 真实遥操里这种情况应该「打滑」：像离合器打滑一样，人手可以继续动，
            # 臂停在它能到的地方；等人手回到可达区域，从新基准继续跟。
            # 绝对对应关系会平移一次，但这比抱着一个永远追不上的目标死磕好得多。
            if engaged and ref is not None:
                # 优化式求解器**永不返回 None**，所以 env.stats["unreachable"]
                # 在它底下恒为 0，这张网会整个失效（实测：腕部够不到时 6 帧
                # 就把 j5/j6 顶死，之后没有任何人吭声，臂一直卡着）。
                # 它改用 stats["saturated"] 报同一件事：顶在限位上且跟不上目标。
                # 两个来源加在一起，换哪个求解器这张网都在。
                unreach = env.stats.get("unreachable", 0)
                k_ik = getattr(env.arm, "_kinematics", None)
                if isinstance(k_ik, AR5OptIK):
                    unreach += k_ik.stats["saturated"]
                if unreach > unreach_seen:
                    slip_run += unreach - unreach_seen
                    unreach_seen = unreach
                    if slip_run >= args.slip_after:
                        slip_run = 0
                        slips += 1
                        ref = None             # 下一帧就地重新抓基准
                        try:
                            env._target_pose = None      # 指令位姿重新锚到实测
                        except Exception:
                            pass
                        # 打滑本身是正常的（碰到可达边界就该滑一下），
                        # 但**连续密集打滑**说明臂是真卡住了 —— 重锚之后
                        # 连它自己当前的位姿都解不出来。那种情况打滑没用，
                        # 只有关节空间回位能出来。
                        now = time.time()
                        if now - slip_t0 < 3.0:
                            slip_burst += 1
                        else:
                            slip_burst = 1
                        slip_t0 = now
                        if slip_burst >= 5:
                            if not slip_stuck_warned:
                                slip_stuck_warned = True
                                print("\n[打滑] 3 秒内连续打滑 5 次 —— 臂是真卡住了"
                                      "（重锚后连自己当前位姿都解不出）。"
                                      "\n       **按 h 关节空间回位**，笛卡尔指令救不回来。")
                        elif now - slip_msg_t >= 2.0:
                            slip_msg_t = now
                            slip_stuck_warned = False
                            print(f"\n[打滑] 臂到不了这个位姿（限位/奇异），已就地重锚 —— "
                                  f"手可以继续动，对应关系平移了一次")
                else:
                    slip_run = max(0, slip_run - 1)
            if stale:
                stale_steps += 1

            # **未接合就不要下发。** 以前这里无条件 step()，即使 Δ=0：
            # step() 仍会拿当前位姿去解一次 IK，而臂停在奇异/限位上时那次 IK
            # 本来就解不出来 —— 于是 50Hz 刷「IK failed」，人还没碰离合就在刷。
            # 手是另一条路（hand.post），不经过 step，所以跳过这里不影响手跟随。
            # 目标断流不会 latch：驱动只在「臂正在动」时才因断流 latch，
            # 停着的时候断流是正常的暂停（rokae_arm.py:434 的注释）。
            # 松手之后不能立刻停发目标：**臂这时候还在动**（还在朝最后的目标伺服），
            # 而 RT 发送器的规则是「正在动的时候目标断流 = latch」
            # （停着断流才算正常暂停，rokae_arm.py:434）。断流即 latch 的后果是
            # 下一次捏合虽然照发目标，却每帧都被 latch 拒掉，表现就是
            # 「捏合一次之后再捏就没反应」。
            # 所以摘离合后继续发零增量，直到臂真正停下来为止。
            settling = False
            if not engaged:
                try:
                    settling = not env.arm._at_rest()
                except Exception:
                    settling = settle_frames > 0        # 读不到就按帧数兜底
                settle_frames = (settle_frames - 1) if settling else 0
            else:
                settle_frames = int(args.hz * 0.6)      # 接合中随时准备好兜底帧数

            if movel is not None:
                # ── MoveL 后端：跳过 env.step 整条链 ──────────────────────
                # rt 后端是 绝对目标 → delta → env.step → **本地逆解** → 关节流。
                # movel 后端把绝对目标**原样**交给控制器，逆解/规划/限位都在那边，
                # 所以 delta、缰绳、本地逆解这些在这条路上全都不参与。
                # 延迟由 MoveLStreamer 的前导距离控住（250mm/s 实测中位 41ms）。
                if engaged and ref is not None:
                    _rpy = matrix_to_rpy(R_target) if (args.with_rotation
                                                       and R_target is not None) else None
                    movel_res = movel.update(p_target, _rpy)
                elif not engaged and movel_started:
                    # 松手：不清队列（清了下次要多吃一次 42.8ms 的 moveStart），
                    # 停止喂新目标，让它自然走完最后几段
                    movel_res = "idle"
                movel_started = True
            elif engaged or settling:
                action = np.zeros(ACTION_DIM)
                action[:6] = delta                      # 未接合时 delta 恒为 0
                env.step(action)                        # use_hand=False，夹爪维走 NullHand

            # latch 了就必须让人看见：latch 之后每一帧目标都会被拒，
            # 表现是「捏合没反应」，但屏幕上什么都不会说。
            # 控制器级保护停机（抱闸吸合、完全不受控）和软件 latch 是两回事，
            # 后者臂还带电。掉电了就把控制器自己的说法问出来。
            if steps % max(1, int(args.hz)) == 0 and not args.mock:
                try:
                    if str(env.arm._robot.powerState({})) .endswith("off") and not power_warned:
                        power_warned = True
                        print("\n[掉电] 控制器已断电（听到的抱闸声就是它）。")
                        diagnose_stop(env, "掉电诊断")
                except Exception:
                    pass

            try:
                if env.arm.latched and not latch_warned:
                    latch_warned = True
                    print("\n[latch] 臂已自锁，所有目标都会被拒绝。"
                          "按 ENTER 摘再按 ENTER 挂（会自动解锁），或按 h 回位。")
                elif not env.arm.latched:
                    latch_warned = False
            except Exception:
                pass

            # ── 关节限位死循环的检测与自救 ──────────────────────────────
            # vel_ar5 里两套限位对不上：kinematics.ik 收敛后用**不含 margin**
            # 的原始限位 np.clip(q, joint_min, joint_max) 夹一下就当有效解返回，
            # 而 rokae_arm.servo_to_joints 拿 **含 margin(0.05)** 的带子判。
            # 于是 IK 会反复交出一个结构上永远通不过的解（j5=-0.8727 vs ±0.8227）。
            # 又因为 _base_pose() 的重锚判据只看平移，纯姿态卡住时 drift 一直很小、
            # 永不重锚 —— 零增量也出不来，只能 50Hz 刷屏。
            # 这里自己兜：连续被拒就把指令位姿清掉，下一步从**实测位姿**重新积分。
            try:
                rej = env.arm.rt_stats()["rejected"].get("limits", 0)
            except Exception:
                rej = limit_rej_seen
            if rej > limit_rej_seen:
                limit_rej_run += rej - limit_rej_seen
                limit_rej_seen = rej
                if limit_rej_run >= args.relimit_after:
                    limit_rej_run = 0
                    env._target_pose = None          # 下一次 _base_pose() 回到实测
                    stuck_recoveries += 1
                    q, _m, _k = joint_margins(env)
                    lo, hi = soft_limits(env)
                    bad = [f"j{i}={q[i]:+.4f}∉[{lo[i]:.4f},{hi[i]:.4f}]"
                           for i in range(len(q)) if not (lo[i] <= q[i] <= hi[i])]
                    print(f"\n[自救] 指令连续被限位拒绝 -> 已重锚到实测位姿")
                    if bad:
                        print(f"  ⚠ 但**实测关节本身**已在软限位外: {', '.join(bad)}")
                        print("     笛卡尔指令救不回来（每个解都会被拒）。"
                              "**按 h 关节空间回位**，那条路不用 IK。")
                    else:
                        print(f"  实测关节都在带内，最紧的是 j{_k} 余量 "
                              f"{np.rad2deg(_m[_k]):.1f}° —— 往回退一点再继续。")
            else:
                limit_rej_run = max(0, limit_rej_run - 1)   # 恢复正常就慢慢清账

            steps += 1
            if steps % 25 == 0:                 # 主循环真实频率
                _now = time.time()
                loop_hz = 25.0 / max(_now - _hz_t0, 1e-6)
                _hz_t0 = _now
            cmd_travel += float(np.linalg.norm(delta[:3]))
            rot_dropped_sum += dropped

            if print_every and steps % print_every == 0:
                ee = env.arm.get_ee_pose()[:3]
                # 预警：哪根关节快撞限位了。撞上去之前就该看见
                tight_note = ""
                try:
                    _q, m, k = joint_margins(env)
                    if np.rad2deg(m[k]) < args.warn_joint_margin:
                        tight_note = f" ⚠j{k}余{np.rad2deg(m[k]):+.1f}°"
                except Exception:
                    pass
                flag = ("接合" if engaged else
                        ("链路陈旧" if stale else
                         ("等右手捏合" if armed else "未挂离合·按ENTER")))
                meas = hand.measured() if hand is not None else None
                hs = ("手" + ",".join(f"{a:4d}" for a in angles) +
                      ("" if meas is None else " 实" + ",".join(f"{m:4d}" for m in meas)))
                sys.stdout.write(
                    # L捏 显示**离拇指最近的那根手指**的距离——正在捏的就是它
                    f"\r[{flag:^12}] R捏={r_pinch:.3f} L捏={float(np.min(pinch_d)):.3f} "
                    f"末端({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f}) "
                    f"Δ={np.linalg.norm(delta[:3])*1000:4.1f}mm "
                    f"Δrot={np.rad2deg(np.linalg.norm(delta[3:6])):4.1f}°"
                    + (f"(丢{np.rad2deg(dropped):4.1f}°)" if dropped > 1e-4 else "")
                    + f" {hs}{rot_held}{tight_note}"
                    + (f"  手转({rot_cmp[0][0]:+.0f},{rot_cmp[0][1]:+.0f},{rot_cmp[0][2]:+.0f})"
                       f"末端转({rot_cmp[1][0]:+.0f},{rot_cmp[1][1]:+.0f},{rot_cmp[1][2]:+.0f})"
                       if rot_cmp is not None else "")
                    + (f"  [MoveL {movel_res}]" if movel is not None else "")
                    + f"  [臂{loop_hz:.0f}Hz 手{tracker.hz_est:.0f}Hz]   ")
                sys.stdout.flush()
                if args.hand_debug:
                    print("\n      " + tracker.debug_line())

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        if old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        print("\n收尾 ...")
        # 每段独立 try：挤在一个 try 里的话，前一步抛异常会把后面的清理全跳过
        try:
            tracker.stop(); tracker.join(timeout=1.0)
        except Exception:
            pass
        if hand is not None:
            try:
                hand.stop_writer()
                hand.set_angles([1000] * NUM_DOF)   # 张开，别让手夹着东西过夜
            except Exception as e:
                print(f"  张手失败: {type(e).__name__}: {e}")
            try:
                print("  手状态:\n" + hand.diagnose())
            except Exception as e:
                print(f"  诊断失败: {type(e).__name__}: {e}")
            try:
                hand.close()
            except Exception as e:
                print(f"  关手失败: {type(e).__name__}: {e}")
        try:
            # movel 后端下 RT 流已经停了，env.step 会踩空（servo_to_pose_rt 要求 RT 在跑）
            if movel is None:
                env.step(np.zeros(ACTION_DIM))
        except Exception as e:
            print(f"  归零失败: {type(e).__name__}: {e}")
        try:
            env.stop()
        except Exception as e:
            print(f"  停臂失败: {type(e).__name__}: {e}")
        try:
            env.close()
        except Exception as e:
            print(f"  断开失败: {type(e).__name__}: {e}")
        try:
            k = getattr(env.arm, "_kinematics", None)
            if isinstance(k, AR5OptIK) and k.stats["calls"]:
                st = k.stats
                n = max(st["calls"], 1)
                print(f"  IK: 调用 {st['calls']}  给解 {st['ok']} ({100*st['ok']/n:.1f}%)  "
                      f"其中打滑 {st['slipped']}（够不到，顶在边界上，**不是错误**）  "
                      f"顶步长上限 {st['stepcap']}（臂在全速追）  丢帧 {st['rejected']}  "
                      f"平均 {1000*st['seconds']/n:.2f} ms/次")
                if st["singular"]:
                    print(f"  ⚠ 奇异闸拦下 {st['singular']} 帧 —— 逆解要往奇异位形里钻，"
                          f"已就地拦住。那里伺服跟不上，力矩尖峰会被碰撞检测判成撞了 → 抱死。")
            elif isinstance(k, AR5IK) and k.stats["calls"]:
                st = k.stats
                print(f"  IK: 调用 {st['calls']}  成功 {st['ok']}  "
                      f"早退 {st['stalled']}  超限 {st['limits']}  "
                      f"平均迭代 {st['iters']/max(st['calls'],1):.1f}  "
                      f"加阻尼 {st['damped']} 次")
        except Exception:
            pass
        if jit is not None:
            print("\n抖动逐级定位（高频能量 = 二阶差分的 p95，越大越抖）：")

            def _hf(a, scale, unit):
                a = np.asarray(a, dtype=float)
                if len(a) < 4:
                    return f"  样本不足({len(a)})"
                d2 = np.abs(np.diff(a, n=2, axis=0))
                return (f"p95 {np.nanpercentile(d2, 95) * scale:8.4f}{unit}   "
                        f"max {np.nanmax(d2) * scale:8.4f}{unit}")
            print(f"  ① 头显原始位置      {_hf(jit['raw'], 1000, 'mm')}")
            print(f"  ② 滤波之后          {_hf(jit['filt'], 1000, 'mm')}")
            print(f"  ③ IK 关节指令       {_hf(jit['qcmd'], 180 / np.pi, '°')}")
            print(f"  ④ 实测关节          {_hf(jit['qact'], 180 / np.pi, '°')}")
            print("  读法：①≫② = 滤波在起作用；②大 = 头显本身噪声大，调小 --arm-smooth；")
            print("        ②小而③大 = 抖在逆解（换 --ik / 调 --ik-max-step）；")
            print("        ③小而④大 = 抖在臂的伺服/机械，和上位机无关。")
        if movel is not None:
            try:
                print(f"  {movel.report()}")
                movel.stop()
                print("  [MoveL] 已清队列")
            except Exception as exc:                    # noqa: BLE001
                print(f"  [MoveL] 收尾: {type(exc).__name__}: {exc}")
        if not args.mock:
            diagnose_stop(env, "退出时状态")
        print(f"  步数 {steps}  接合 {engaged_steps}  陈旧 {stale_steps}  "
              f"姿态限幅 {rot_held_n}  打滑 {slips}  限位自救 {stuck_recoveries}  "
              f"指令行程 {cmd_travel*1000:.0f} mm")
        if rot_dropped_sum > 1e-3:
            print(f"  rot-mode={args.rot_mode} 累计丢弃姿态 "
                  f"{np.rad2deg(rot_dropped_sum):.0f}°"
                  f"（roll 模式只保留绕工具轴的转动，上下挥手会被丢掉；要上下就用 --rot-mode full）")
        try:
            rs = env.arm.rt_stats()
            print(f"  臂 RT: 已发 {rs['sent']}  迟到 {rs['late_ticks']}  "
                  f"限位命中 {rs['limit_hits']}  latch={rs['latched']}")
            rej = {k: v for k, v in rs["rejected"].items() if v}
            if rej:
                print(f"  臂拒绝的目标: {rej}   (limits=IK解不出/超关节限位)")
        except Exception:
            pass
        try:
            print("\n手的实际量程（整场 min~max）：")
            print(tracker.range_report())
        except Exception:
            pass
        if hand is not None:
            print(f"  手: 写 {hand.writes} 次, I/O 错误 {hand.io_errors}, 钳位 {hand.clamped}")
            if hand.writes == 0:
                print("  ⚠ **一次都没下发** —— 手不动不是手的问题，是上游没给出指令。"
                      "看下面的「重定向出错」和 --hand-debug 的 raw=")
        else:
            print("  手: 【本次没有接手】—— 启动参数里有 --no-hand（或 --mock 没配 "
                  "--real-hand）。想让手动，去掉它。")
        try:
            if tracker.errors:
                print(f"  ⚠ 手的重定向出错 {tracker.errors} 次，最后一条: {tracker.last_error}")
        except Exception:
            pass
        for k, v in env.stats.items():
            if v:
                print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
