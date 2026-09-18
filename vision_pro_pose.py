"""
Vuer 端脚本：读取 Apple Vision Pro 的头部位姿 + 双手位姿（同一局域网）。

架构说明
--------
Vuer 的 Python 端其实是「服务器」，Vision Pro 上的 Safari 是「客户端」：
    Vision Pro (Safari, WebXR)  --wss-->  本脚本 (Python, Vuer)
本脚本收到两类事件：
    CAMERA_MOVE : 头显相机（=头部）位姿，4x4 齐次矩阵
    HAND_MOVE   : 左右手各 25 个关节的 4x4 齐次矩阵 + 捏合/握拳状态

运行
----
    python gen_cert.py            # 只需一次，生成 cert.pem / key.pem
    python vision_pro_pose.py     # 启动，终端会打印 Network 地址
在 Vision Pro 的 Safari 打开  https://<本机局域网IP>:8012
（自签证书首次会提示不受信任，点“显示详细信息 -> 访问此网站”），
然后点页面右下角的 "Enter VR / AR" 按钮，进入沉浸模式后就开始推流。
"""

import asyncio
import json
import time
from pathlib import Path

import numpy as np
from msgpack import ExtType
from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, Hands

# ============ 配置 ============
HOST = "0.0.0.0"          # 监听所有网卡，允许局域网访问
PORT = 8012
CERT = "cert.pem"         # 由 gen_cert.py 生成；WebXR 必须 HTTPS
KEY = "key.pem"
PRINT_HZ = 5              # 终端打印频率（数据本身按设备帧率 ~60-90Hz 到达）
LOG_FILE = None           # 例如 "pose_log.jsonl"，为 None 则不落盘
# ==============================

# XRHand 关节顺序（与 Vuer / WebXR 一致，共 25 个）
HAND_JOINTS = [
    "wrist",
    "thumb-metacarpal", "thumb-phalanx-proximal", "thumb-phalanx-distal", "thumb-tip",
    "index-finger-metacarpal", "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate", "index-finger-phalanx-distal", "index-finger-tip",
    "middle-finger-metacarpal", "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate", "middle-finger-phalanx-distal", "middle-finger-tip",
    "ring-finger-metacarpal", "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate", "ring-finger-phalanx-distal", "ring-finger-tip",
    "pinky-finger-metacarpal", "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate", "pinky-finger-phalanx-distal", "pinky-finger-tip",
]


class PoseState:
    """保存最新位姿，供你自己的代码（机器人控制等）读取。"""

    def __init__(self):
        self.head = None            # 4x4 np.ndarray（世界系 -> 头部）
        self.left = None            # (25, 4, 4)
        self.right = None           # (25, 4, 4)
        self.left_state = {}
        self.right_state = {}
        self.t_head = 0.0
        self.t_hand = 0.0
        self.n_head = 0
        self.n_hand = 0


state = PoseState()
_log_fp = open(LOG_FILE, "a", encoding="utf-8") if LOG_FILE else None


# msgpackr 的 TypedArray 类型索引 -> numpy dtype（小端）
_MSGPACKR_TYPED_ARRAYS = {
    0: "<i1", 1: "<u1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4",
    7: "<f4", 8: "<f8", 9: "<i8", 10: "<u8",
}


def to_float_array(data) -> np.ndarray:
    """把客户端传来的数据统一转成一维 float 数组。

    前端的 Float32Array 经 msgpack 传输后可能是：
      - bytes / bytearray / memoryview（原始 float32 小端字节流）
      - 由单字节 bytes 组成的 list
      - 普通 float list
    """
    if isinstance(data, ExtType):
        # 前端 msgpackr 对 TypedArray 的编码：ext code 116，data[0] 是类型索引，其后是原始字节
        if data.code == 116 and len(data.data) >= 1:
            dtype = _MSGPACKR_TYPED_ARRAYS.get(data.data[0])
            if dtype is None:
                raise TypeError(f"未知 TypedArray 类型码 {data.data[0]}")
            return np.frombuffer(bytes(data.data[1:]), dtype=dtype).astype(np.float64)
        # 其他扩展类型：按 float32 字节流兜底
        return np.frombuffer(bytes(data.data), dtype="<f4").astype(np.float64)
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(data), dtype="<f4").astype(np.float64)
    if isinstance(data, np.ndarray):
        return data.astype(np.float64).ravel()
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and isinstance(data[0], (bytes, bytearray)):
            return np.frombuffer(b"".join(data), dtype="<f4").astype(np.float64)
        return np.asarray(data, dtype=np.float64).ravel()
    if isinstance(data, dict):  # 某些序列化会包成 {"data": [...]} 或 {"0":..,"1":..}
        if "data" in data:
            return to_float_array(data["data"])
        return np.asarray([data[k] for k in sorted(data, key=int)], dtype=np.float64)
    raise TypeError(f"无法解析的数据类型: {type(data)}")


def col_major_to_mat(flat) -> np.ndarray:
    """WebGL/WebXR 的 16 元素列主序数组 -> 4x4 numpy 矩阵。"""
    return to_float_array(flat).reshape(4, 4, order="F")


def mat_to_pos_quat(T: np.ndarray):
    """4x4 -> (xyz, 四元数 wxyz)。"""
    pos = T[:3, 3]
    R = T[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return pos, np.array([w, x, y, z])


def _log(kind, payload):
    if _log_fp is None:
        return
    _log_fp.write(json.dumps({"t": time.time(), "type": kind, **payload}) + "\n")


# ---------------------------------------------------------------------------
app = Vuer(
    host=HOST,
    port=PORT,
    cert=CERT,
    key=KEY,
    # 使用 vuer 自带的本地前端，Vision Pro 直接访问 https://<本机IP>:8012，不经过 vuer.ai
    client_url="https://{local_ip}:{port}",
)


@app.add_handler("CAMERA_MOVE")
async def on_camera(event, session: VuerSession):
    """头部（相机）位姿。event.value 通常形如 {"camera": {"matrix": [...16], ...}}。"""
    v = event.value or {}
    cam = v.get("camera", v)
    flat = cam.get("matrix")
    if flat is None:
        return
    try:
        arr = to_float_array(flat)
    except Exception as e:
        print("CAMERA_MOVE 解析失败:", e, type(flat))
        return
    if arr.size != 16:
        return
    state.head = arr.reshape(4, 4, order="F")
    state.t_head = time.time()
    state.n_head += 1
    _log("head", {"matrix": arr.tolist()})


@app.add_handler("HAND_MOVE")
async def on_hand(event, session: VuerSession):
    """双手位姿。event.value = {left: 25*16, right: 25*16, leftState, rightState}。"""
    v = event.value or {}
    log_payload = {}
    for side in ("left", "right"):
        flat = v.get(side)
        if flat is not None:
            try:
                arr = to_float_array(flat)
            except Exception as e:
                print(f"HAND_MOVE[{side}] 解析失败:", e, type(flat))
                arr = None
            if arr is not None and arr.size == 25 * 16:
                # 每个关节 16 个数（列主序）-> (25,4,4)
                mats = arr.reshape(25, 4, 4).transpose(0, 2, 1)
                setattr(state, side, mats)
                log_payload[side] = arr.tolist()
            elif arr is not None:
                print(f"HAND_MOVE[{side}] 长度异常: {arr.size}（期望 400）")
        st = v.get(f"{side}State")
        if st is not None:
            setattr(state, f"{side}_state", st)
            log_payload[f"{side}State"] = st
    state.t_hand = time.time()
    state.n_hand += 1
    _log("hands", log_payload)


@app.spawn(start=True)
async def main(session: VuerSession):
    # 场景里放一个 Hands 组件并开启 stream，才会推送 HAND_MOVE；相机移动自动推送 CAMERA_MOVE。
    # grid=False：AR 透视模式下不画虚拟地板，只叠加手部骨架，方便看真实机械臂。
    session.set @ DefaultScene(
        Hands(key="hands", stream=True),
        grid=False,
    )
    print("客户端已连接，请在 Vision Pro 上点右下角 Enter AR（透视模式，可看到真实机械臂）...")

    last_print = 0.0
    while True:
        await asyncio.sleep(0.01)
        now = time.time()
        if now - last_print < 1.0 / PRINT_HZ:
            continue
        last_print = now

        lines = []
        if state.head is not None:
            p, q = mat_to_pos_quat(state.head)
            lines.append(
                f"[HEAD] pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) "
                f"quat(wxyz)=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f}) n={state.n_head}"
            )
        for side in ("left", "right"):
            mats = getattr(state, side)
            if mats is None:
                continue
            wrist_p, _ = mat_to_pos_quat(mats[0])
            idx_tip_p, _ = mat_to_pos_quat(mats[HAND_JOINTS.index("index-finger-tip")])
            st = getattr(state, f"{side}_state") or {}
            lines.append(
                f"[{side.upper():5s}] wrist=({wrist_p[0]:+.3f},{wrist_p[1]:+.3f},{wrist_p[2]:+.3f}) "
                f"indexTip=({idx_tip_p[0]:+.3f},{idx_tip_p[1]:+.3f},{idx_tip_p[2]:+.3f}) "
                f"pinch={st.get('pinch', False)} squeeze={st.get('squeeze', False)} n={state.n_hand}"
            )
        if lines:
            print("\n".join(lines) + "\n")


if __name__ == "__main__":
    if not (Path(CERT).exists() and Path(KEY).exists()):
        raise SystemExit(
            f"找不到 {CERT}/{KEY}。WebXR 必须走 HTTPS，请先运行:  python gen_cert.py"
        )
    app.run()
