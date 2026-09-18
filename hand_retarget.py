#!/usr/bin/env python3
"""Vision Pro 25 关节手部骨架  ->  因时 RH56E2 六自由度角度。

头显给的是每个关节相对手腕的 4x4 矩阵，因时要的是 6 个无量纲量程值
（1000=张开 0=握紧）。中间隔着「人手 25 自由度 -> 机械手 6 自由度」的降维，
没有解析解，只能取每根手指的**弯曲标量**再线性映射。

关节编号（avp_stream/streamer.py:82，权威）:
    0 wrist
    1-4   拇指  thumbKnuckle / IntermediateBase / IntermediateTip / Tip
    5-9   食指  Metacarpal / Knuckle / IntermediateBase / IntermediateTip / Tip
    10-14 中指   15-19 无名指   20-24 小指
    (25-26 前臂，新版固件才有，用不到)

弯曲标量取法
------------
四指：knuckle -> interBase -> interTip -> tip 这三段之间的两个转折角之和。
用**角度**而不是「指尖到手腕的距离」，因为角度与手的大小无关，换个人不用重标。

拇指弯曲：同法，用 knuckle(1) -> ... -> tip(4) 的两个转折角。
拇指旋转：拇指轴与掌面横轴（食指根 -> 小指根）的夹角。这个量是否单调、
量程多大，**因人而异**，所以端点一律由标定确定，不写死。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from inspire_hand6 import (DEFAULT_LIMITS, DOF_NAMES, INDEX, LITTLE, MIDDLE,
                           NUM_DOF, RING, THUMB_BEND, THUMB_ROT)

WRIST = 0
# 每根手指参与算弯曲的关节链（从掌指关节往指尖）
CHAINS = {
    THUMB_BEND: (1, 2, 3, 4),
    INDEX:      (6, 7, 8, 9),
    MIDDLE:     (11, 12, 13, 14),
    RING:       (16, 17, 18, 19),
    LITTLE:     (21, 22, 23, 24),
}
KNUCKLE = {INDEX: 6, MIDDLE: 11, RING: 16, LITTLE: 21}
THUMB_KNUCKLE, THUMB_TIP = 1, 4
INDEX_TIP = 9
TIPS = {INDEX: 9, MIDDLE: 14, RING: 19, LITTLE: 24}
# 会跟拇指做对捏的四根手指，顺序固定
PINCH_FINGERS = (INDEX, MIDDLE, RING, LITTLE)

# 捏合距离端点（米）的兜底值：指尖相触约 1.5cm；张开时四根离拇指的距离不同，
# 小指最远。同样建议现场标定，手大小差异不小。
FALLBACK_PINCH_OPEN = [0.090, 0.105, 0.115, 0.120]   # 食指/中指/无名指/小指
FALLBACK_PINCH_CLOSED = 0.015

# 捏合驱动只在「最近的这一段」生效：指尖距离归一化到 0(相触)~1(张开)，
# 小于这个比例才算「在捏」。标定的张开端点是手指完全张开量的，
# 遥操时手远没那么张，铺满整个区间会把正常张开误判成半捏。
PINCH_GATE = 0.45

# 没有标定文件时的兜底端点（弧度）。只是让程序能跑起来，
# **务必现场跑一次 --calib**，人手差异比想象的大。
FALLBACK = {
    LITTLE:     (0.20, 2.90),
    RING:       (0.20, 2.90),
    MIDDLE:     (0.20, 2.90),
    INDEX:      (0.20, 2.90),
    THUMB_BEND: (0.15, 1.70),
    THUMB_ROT:  (1.45, 0.65),   # 注意：旋转这一维 open 端的原始值比 closed 端**大**
}


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(float(a @ b) / (na * nb), -1.0, 1.0)))


def raw_features(fingers: np.ndarray) -> np.ndarray:
    """(25,4,4) 关节矩阵 -> 6 个原始标量（弧度）。

    fingers 是**相对手腕**的矩阵，直接取平移即可，不必乘回世界系：
    弯曲角是手内部的相对量，跟手腕在哪无关。
    """
    p = np.asarray(fingers)[:, :3, 3]        # (25,3) 各关节位置
    out = np.zeros(NUM_DOF)

    for dof, chain in CHAINS.items():
        j0, j1, j2, j3 = chain
        s1, s2, s3 = p[j1] - p[j0], p[j2] - p[j1], p[j3] - p[j2]
        out[dof] = _angle(s1, s2) + _angle(s2, s3)

    # 拇指旋转：拇指轴 vs 掌面横轴（食指根 -> 小指根）
    across = p[KNUCKLE[LITTLE]] - p[KNUCKLE[INDEX]]
    thumb_axis = p[THUMB_TIP] - p[THUMB_KNUCKLE]
    out[THUMB_ROT] = _angle(thumb_axis, across)
    return out


def pinch_distances(fingers: np.ndarray) -> np.ndarray:
    """拇指尖到**四根手指**指尖的距离（米），顺序同 PINCH_FINGERS。

    为什么单独要这个量：人做捏取时手指只是中等弯曲（主要弯在掌指关节），
    拇指几乎不弯——把两个指尖带到一起的是**对掌**，不是弯曲。所以单看
    「弯曲角之和」会判成「才半闭」，机械手就停在离得很远的地方，
    而握拳因为真的把每节都弯到底，反而没问题。

    四根都算，是因为人不只用食指捏：拇指对中指、对无名指都是常用的捏法。
    """
    p = np.asarray(fingers)[:, :3, 3]
    return np.array([float(np.linalg.norm(p[THUMB_TIP] - p[TIPS[f]])) for f in PINCH_FINGERS])


def pinch_distance(fingers: np.ndarray) -> float:
    """兼容旧调用：拇指到食指的距离。"""
    return float(pinch_distances(fingers)[0])


@dataclass
class Calibration:
    """每个自由度两个端点：原始值 -> 机械手角度。

    open 端映射到该维安全带的上界（张开），closed 端映射到下界（握紧）。
    下界不是 0 —— 食指和拇指会互相顶死，见 inspire_hand6.DEFAULT_LIMITS。
    """
    raw_open: list = field(default_factory=lambda: [FALLBACK[i][0] for i in range(NUM_DOF)])
    raw_closed: list = field(default_factory=lambda: [FALLBACK[i][1] for i in range(NUM_DOF)])
    limits: dict = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    # 捏合距离端点（米）：张开时四根手指各自到拇指的距离 / 指尖相触时（四根共用）
    pinch_open: list = field(default_factory=lambda: list(FALLBACK_PINCH_OPEN))
    pinch_closed: float = FALLBACK_PINCH_CLOSED

    def pinch_angles(self, d4, gate: float = PINCH_GATE) -> Optional[dict]:
        """四个指尖距离 -> {自由度: 角度}。距离越小越闭合。

        拇指弯曲由**距离最近的那根手指**决定——谁在跟拇指捏，拇指就配合谁。

        `gate`：捏合驱动**只在最近的那一段距离里生效**，不铺满整个区间。
        为什么必须这样——标定的「张开」端点是手指**完全张开**时量的
        （实测拇指到小指 18.9cm），而遥操时手很少张到那个程度，正常大概 10cm。
        铺满区间的话，10cm 会被算成 t=0.5 → 小指被压到 500（半闭），
        再和弯曲角取 min 就一直松不开 —— 表现就是「小指恒弯」。
        加了 gate 之后，t >= gate 时捏合项给满量程 1000，取 min 等于不参与，
        完全交给弯曲角；t < gate 才线性介入，且在 t = gate 处连续。
        """
        d4 = np.atleast_1d(np.asarray(d4, dtype=float))
        gate = float(np.clip(gate, 0.05, 1.0))
        out = {}
        ts = []
        for k, dof in enumerate(PINCH_FINGERS):
            span = self.pinch_open[k] - self.pinch_closed
            if span < 1e-6:
                continue
            t = float(np.clip((d4[k] - self.pinch_closed) / span, 0.0, 1.0))  # 0=相触 1=张开
            ts.append(t)
            tg = min(1.0, t / gate)                 # gate 之外一律视作「完全没在捏」
            lo, hi = self.limits[dof]
            out[dof] = int(round(lo + tg * (hi - lo)))
        if not ts:
            return None
        lo, hi = self.limits[THUMB_BEND]
        out[THUMB_BEND] = int(round(lo + min(1.0, min(ts) / gate) * (hi - lo)))
        return out

    def to_angles(self, raw: np.ndarray) -> list:
        out = []
        for i in range(NUM_DOF):
            lo, hi = self.limits[i]
            span = self.raw_closed[i] - self.raw_open[i]
            if abs(span) < 1e-6:                 # 标定塌缩，这一维就不动
                out.append(int(round((lo + hi) / 2)))
                continue
            t = (float(raw[i]) - self.raw_open[i]) / span      # 0=张开 1=握紧
            t = float(np.clip(t, 0.0, 1.0))
            out.append(int(round(hi + t * (lo - hi))))
        return out

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "raw_open": [float(v) for v in self.raw_open],
            "raw_closed": [float(v) for v in self.raw_closed],
            "pinch_open": [float(v) for v in np.atleast_1d(self.pinch_open)],
            "pinch_closed": float(self.pinch_closed),
            "pinch_fingers": [DOF_NAMES[f] for f in PINCH_FINGERS],
            "limits": {str(k): list(v) for k, v in self.limits.items()},
            "dof_names": DOF_NAMES,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Calibration":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        ro, rc = d["raw_open"], d["raw_closed"]
        if len(ro) != NUM_DOF or len(rc) != NUM_DOF:
            raise ValueError(f"标定文件维度不对: {len(ro)}/{len(rc)}, 期望 {NUM_DOF}")
        if not all(np.isfinite(ro)) or not all(np.isfinite(rc)):
            # 另一个项目就栽在 NaN 悄悄写进标定文件、所有守卫对 NaN 全部失效上
            raise ValueError(f"标定文件里有非有限值: open={ro} closed={rc}")
        lim = {int(k): tuple(v) for k, v in d.get("limits", {}).items()} or dict(DEFAULT_LIMITS)
        po = d.get("pinch_open", FALLBACK_PINCH_OPEN)
        po = [float(v) for v in np.atleast_1d(po)]
        if len(po) == 1:                     # 旧格式：只有食指一个端点，其余按兜底
            po = [po[0]] + list(FALLBACK_PINCH_OPEN[1:])
        if len(po) != len(PINCH_FINGERS):
            raise ValueError(f"pinch_open 长度 {len(po)}，期望 {len(PINCH_FINGERS)}")
        pc = float(d.get("pinch_closed", FALLBACK_PINCH_CLOSED))
        if not (all(np.isfinite(po)) and np.isfinite(pc)):
            raise ValueError(f"捏合端点非有限: {po}/{pc}")
        return cls(raw_open=ro, raw_closed=rc, limits=lim, pinch_open=po, pinch_closed=pc)


class Retargeter:
    """原始特征 -> 平滑 -> 机械手角度。"""

    def __init__(self, calib: Optional[Calibration] = None, smooth: float = 0.35,
                 use_pinch: bool = True, pinch_gate: float = PINCH_GATE):
        self.calib = calib or Calibration()
        self.smooth = float(np.clip(smooth, 0.0, 0.95))
        self.use_pinch = use_pinch
        self.pinch_gate = pinch_gate
        self._ema: Optional[np.ndarray] = None
        self._ema_d: Optional[float] = None

    def __call__(self, fingers: np.ndarray) -> tuple:
        raw = raw_features(fingers)
        if not np.all(np.isfinite(raw)):
            raw = np.nan_to_num(raw, nan=0.0)
        d4 = pinch_distances(fingers)
        self._ema = raw if self._ema is None else (
            self.smooth * self._ema + (1.0 - self.smooth) * raw)
        self._ema_d = d4 if self._ema_d is None else (
            self.smooth * self._ema_d + (1.0 - self.smooth) * d4)

        angles = self.calib.to_angles(self._ema)
        curl_only = list(angles)          # 留一份纯弯曲角，调试时对比用
        pin = None

        # 捏合覆盖：弯曲量和指尖距离各给一个答案，**取更闭合的那个**。
        # 握拳时弯曲量赢（指尖距离反而不小），捏取时指尖距离赢（弯曲量看着才半闭）。
        if self.use_pinch:
            pin = self.calib.pinch_angles(self._ema_d, self.pinch_gate)
            if pin is not None:
                for dof, v in pin.items():
                    angles[dof] = min(angles[dof], v)
        self.last_curl = curl_only
        self.last_pinch = pin
        return angles, raw, self._ema_d

    def reset(self) -> None:
        self._ema = None
        self._ema_d = None


# ── 手掌坐标系：直接从骨架建，不碰 wrist 矩阵 ────────────────────────────────
def palm_frame(fingers: np.ndarray) -> np.ndarray:
    """从 25 关节骨架建一个**右手**正交系，返回 3x3（列 = x,y,z 轴，在手腕系下）。

        x = 手腕 → 中指指根        「手指朝向」
        z = 掌法向（从**手背**穿出）
        y = z × x                  补齐右手系

    为什么不用 `left_wrist` 的旋转矩阵：visionOS 的左右手腕系约定**不一样**
    （avp_stream 的 trn_constants 里 VISIONOS_LEFT_HAND_TO_LEAP 和
    RIGHT 差一个 ±90°绕Y），照抄一只手的约定用到另一只手上，表现就是
    姿态整体拧了 90°、手心手背分不出来。

    骨架没有这个问题：三个掌骨点定义的平面是解剖学上的手掌平面，和厂商
    怎么定义手腕系无关。而且这里用**显式叉乘**正交化，det 恒为 +1 ——
    左手数据也绝不会退化成镜像（det=-1），这是镜像问题的根上。

    fingers: (25,4,4)，每个关节**相对手腕**的位姿（avp_stream 的 left_fingers）。
    """
    f = np.asarray(fingers, dtype=np.float64)
    p = f[:, :3, 3]
    wrist = p[WRIST]
    x = p[KNUCKLE[MIDDLE]] - wrist                     # 朝指根
    across = p[KNUCKLE[LITTLE]] - p[KNUCKLE[INDEX]]    # 食指指根 → 小指指根

    nx = float(np.linalg.norm(x))
    if nx < 1e-9:
        return np.eye(3)
    x = x / nx
    across = across - x * float(across @ x)            # 去掉沿 x 的分量
    na = float(np.linalg.norm(across))
    if na < 1e-9:
        return np.eye(3)
    across = across / na
    z = np.cross(x, across)                            # 掌法向
    nz = float(np.linalg.norm(z))
    if nz < 1e-9:
        return np.eye(3)
    z = z / nz
    y = np.cross(z, x)                                 # 显式补齐，保证右手系
    return np.column_stack([x, y, z])


def palm_normal_world(fingers: np.ndarray, wrist_R: np.ndarray) -> np.ndarray:
    """掌法向在世界系里的单位向量。手指矩阵是相对手腕的，所以要左乘 wrist_R。"""
    return np.asarray(wrist_R, dtype=np.float64) @ palm_frame(fingers)[:, 2]


def palm_facing(fingers: np.ndarray, wrist_R: np.ndarray) -> str:
    """把掌法向说成人话。头显系：X=右 Y=前 Z=上。"""
    n = palm_normal_world(fingers, wrist_R)
    k = int(np.argmax(np.abs(n)))
    pos = n[k] > 0
    # z 轴从手背穿出，所以「法向朝上」= 手背朝上 = **手心朝下**
    return [("手心朝左", "手心朝右"),
            ("手心朝后", "手心朝前"),
            ("手心朝下", "手心朝上")][k][0 if pos else 1]
