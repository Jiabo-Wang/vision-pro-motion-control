#!/usr/bin/env python3
"""因时 RH56E2 灵巧手 6 自由度驱动 —— Project_Atom 自用，独立实现。

和 vel_ar5/VEL 那套是两个项目：协议细节参考了 VEL 已验证的实现，但代码不共享、
不 import，两边各自演进。

只依赖 pyserial。

协议要点（都是实物验证过的，别凭手册改）
------------------------------------
串口帧 21 字节（转义前）:
    AA AA | CAN_ID 4B小端 | 数据 8B | DLC | 00 01 00 | CKS | 55 55
    CKS = sum(frame[2:18]) & 0xFF   ← 在**转义之前**算

CAN_ID 是 29 位扩展帧:
    bit 28..26  读写标志  0=读 1=写
    bit 25..14  寄存器字节地址（**只有 12 位，上限 4095**）
    bit 13..0   手 ID

**字节转义**：帧体（下标 2..18）里凡是 0x55 / 0xAA / 0xA5，前面插一个 0xA5。
帧头帧尾不转义，收发双向都要做。官方文档和官方 C++ 参考实现都没提这件事。
不做的后果：含 0xA5 的值静默写不进去；校验和恰好是 0x55 的帧被误判成坏帧。
所以实际帧长是 21~38 不定，不能按固定长度硬对齐。

CAN 数据段上限 8 字节 = 4 个 int16，所以 6 维要拆两帧（第二帧地址 +8）。

自由度顺序（全系列固定）
    0 小指  1 无名指  2 中指  3 食指  4 拇指弯曲  5 拇指旋转
角度是无量纲量程值：1000 = 完全张开，0 = 完全握紧，-1 = 该维保持不动。
"""
from __future__ import annotations

import struct
import threading
import time
from typing import Optional, Sequence

# ── 自由度 ──────────────────────────────────────────────────────────────────
LITTLE, RING, MIDDLE, INDEX, THUMB_BEND, THUMB_ROT = range(6)
DOF_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
NUM_DOF = 6

ANGLE_OPEN, ANGLE_CLOSED, ANGLE_HOLD = 1000, 0, -1

# ── 寄存器（字节地址）──────────────────────────────────────────────────────
ANGLE_SET = 1486
FORCE_SET = 1498
SPEED_SET = 1522
ANGLE_ACT = 1546
FORCE_ACT = 1582
ERROR_REG = 1606
STATUS_REG = 1612
TEMP_REG = 1618

STATUS_MEANING = {
    0: "正在松开", 1: "抓取中", 2: "到位停止", 3: "力控停止",
    5: "电流保护停止", 6: "堵转停止", 7: "未收过指令",
}
ERROR_BITS = {0: "堵转", 1: "过温", 2: "过流", 3: "电机异常", 4: "通信异常"}


def decode_error(code: int) -> str:
    if code == 0:
        return "ok"
    return "|".join(n for b, n in ERROR_BITS.items() if code & (1 << b)) or f"未知(0x{code:02X})"


# ── 安全钳位 ────────────────────────────────────────────────────────────────
# 2026-09-03 在这只手上实测：食指和拇指对向收拢，到「食指 495 / 拇指 273」互相顶死。
# 人手握拳会让跟踪把这两维直接推到 0，那就是持续堵转。下限留在接触点之上。
#
# 拇指旋转会挪动接触点（实测拇指弯曲 400->接触 484，600->370），跟踪旋转时
# 包络会漂，所以这里给的是保守带；真要放宽先用 probe_thumb_rot 现场测。
# 下面食指/拇指的下限直接取 vel_ar5 里**实测验证过**的两指闭合命令值
# （config.py: finger_close=480「接触点 495 再进 15」，thumb_close=500，
#  该组合下食指接触点约 490）。比这更保守的值会让两指永远够不到接触点，
# 结果是「握拳没问题、捏合捏不住」。真正的过载保护是力控阈值，不是这里的钳位。
#
# ⚠ 这些接触点全都是在 **thumb_rot ≈ 100** 下测的。跟踪拇指旋转会让几何
#   整体漂移，做精细捏取时建议 --freeze-thumb-rot。
DEFAULT_LIMITS = {
    LITTLE:     (0, 1000),
    RING:       (0, 1000),
    MIDDLE:     (0, 1000),
    INDEX:      (0, 1000),      # 静态不设限，改用下面的**耦合下限**
    THUMB_BEND: (0, 1000),
    THUMB_ROT:  (60, 400),      # [待核实] 现场没测过安全带
}

# 食指的下限**取决于拇指弯到哪**，不是一个固定值。
#
# 之前给食指钉死 480，代价是：拇指明明张着、两根手指根本碰不到的时候，
# 食指也只能走 1000->480，可见行程只有别的手指一半 —— 表现就是
# 「单独弯食指没什么反应」。而握拳时因为五指一起动，反而看不出来。
#
# 插值点来自 vel_ar5 config.py 的实测记录：
#     拇指 400 -> 接触点 484      拇指 500 -> 接触点 490
#     拇指 600 -> 接触点 370      两指对向时 食指495/拇指273 互顶
# 拇指越张开，两者越碰不到，食指能走得越深；拇指全张时无碰撞风险。
#
# ⚠⚠ 这组数是**捏合**姿态下量的（拇指对着食指指尖），不是握拳姿态。
#    握拳时拇指是压在已经卷好的四指**外面**，压根碰不到食指指尖，
#    这个下限在那里不成立。默认关闭，理由见 couple_index_thumb 的注释。
INDEX_FLOOR_VS_THUMB = [
    (0,    520),   # 拇指全弯：食指必须留在接触点之上
    (400,  500),
    (500,  480),
    (600,  390),
    (800,    0),   # 拇指已经让开
    (1000,   0),
]


def index_floor(thumb_bend: float) -> int:
    """给定拇指弯曲值，食指最低能到哪。"""
    xs = [p[0] for p in INDEX_FLOOR_VS_THUMB]
    ys = [p[1] for p in INDEX_FLOOR_VS_THUMB]
    import bisect
    i = bisect.bisect_left(xs, thumb_bend)
    if i <= 0:
        return ys[0]
    if i >= len(xs):
        return ys[-1]
    t = (thumb_bend - xs[i - 1]) / (xs[i] - xs[i - 1])
    return int(round(ys[i - 1] + t * (ys[i] - ys[i - 1])))


class ProtocolError(RuntimeError):
    pass


class CanCodec:
    """USB-CAN 适配器的串口封装 + 因时 CAN 寄存器读写。"""

    HEAD = b"\xAA\xAA"
    TAIL = b"\x55\x55"
    FRAME_LEN = 21
    BODY_LEN = 17                    # CAN_ID 4 + 数据 8 + DLC 1 + 标志 3 + CKS 1
    MAX_FRAME_LEN = 2 + 2 * 17 + 2   # 帧体全被转义的最坏情况
    MAX_PAYLOAD = 8
    ESC = 0xA5
    SPECIAL = (0x55, 0xAA, 0xA5)
    CMD_READ, CMD_WRITE = 0, 1
    RW_SHIFT, ADDR_SHIFT = 26, 14
    ADDR_MASK, ID_MASK = 0xFFF, 0x3FFF

    @classmethod
    def _escape(cls, frame: bytes) -> bytes:
        body = bytearray()
        for b in frame[2:19]:
            if b in cls.SPECIAL:
                body.append(cls.ESC)
            body.append(b)
        return frame[:2] + bytes(body) + frame[19:]

    @classmethod
    def build(cls, cmd: int, dev_id: int, addr: int, data: bytes, dlc: int) -> bytes:
        if not 0 <= addr <= cls.ADDR_MASK:
            raise ValueError(
                f"寄存器地址 {addr} 超出 12 位 (0..{cls.ADDR_MASK})。"
                "触觉寄存器在 3000~5123，其中 >4095 的部分 CAN 根本寻址不到，要走网口 Modbus TCP。"
            )
        if not 0 <= dev_id <= cls.ID_MASK:
            raise ValueError(f"手 ID {dev_id} 超出 14 位")
        can_id = (cmd << cls.RW_SHIFT) | (addr << cls.ADDR_SHIFT) | dev_id
        f = bytearray(cls.HEAD) + can_id.to_bytes(4, "little")
        f += bytes(data).ljust(8, b"\x00")
        f += bytes([dlc, 0x00, 0x01, 0x00])
        f.append(sum(f[2:]) & 0xFF)          # 校验和算在转义之前
        return cls._escape(bytes(f) + cls.TAIL)

    @classmethod
    def extract(cls, buf: bytes) -> Optional[bytes]:
        """从缓冲里扫出一个合法帧并去转义，返回 21 字节原始帧；扫不到返回 None。"""
        for i in range(len(buf) - cls.FRAME_LEN + 1):
            if buf[i:i + 2] != cls.HEAD:
                continue
            body = bytearray()
            j = i + 2
            while len(body) < cls.BODY_LEN and j < len(buf):
                if buf[j] == cls.ESC:
                    j += 1
                    if j >= len(buf):
                        break
                body.append(buf[j])
                j += 1
            if len(body) < cls.BODY_LEN or buf[j:j + 2] != cls.TAIL:
                continue
            frame = cls.HEAD + bytes(body) + cls.TAIL
            if (sum(frame[2:18]) & 0xFF) == frame[18]:
                return frame
        return None

    @classmethod
    def check(cls, resp: bytes, addr: int, cmd: int) -> None:
        if len(resp) != cls.FRAME_LEN:
            raise ProtocolError(f"帧长 {len(resp)} != {cls.FRAME_LEN}")
        can_id = int.from_bytes(resp[2:6], "little")
        got_addr = (can_id >> cls.ADDR_SHIFT) & cls.ADDR_MASK
        if got_addr != addr:
            raise ProtocolError(f"响应地址 {got_addr} != 请求 {addr}")
        if ((can_id >> cls.RW_SHIFT) & 0x7) != cmd:
            raise ProtocolError("响应读写标志不符")


class InspireHand6:
    """6 维灵巧手。写入走后台线程，绝不阻塞调用方的控制环。

    为什么必须后台写：一次 6 维写要拆 2 帧、单帧往返实测 6~8ms，同步写进
    50Hz 的臂控制环里会把环拖垮（这条是 vel_ar5 那边用血换来的教训：
    手的同步 CAN 写 p99 到过 219ms，臂 250ms 收不到目标就 latch）。
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200,
                 dev_id: int = 1, timeout: float = 0.1, retries: int = 2,
                 limits: Optional[dict] = None, serial_factory=None,
                 read_every: int = 5, couple_index_thumb: bool = False):
        self.port, self.baudrate, self.dev_id = port, baudrate, dev_id
        self.timeout, self.retries = timeout, retries
        self.limits = dict(DEFAULT_LIMITS if limits is None else limits)
        self.read_every = max(1, int(read_every))   # 每写 N 次回读一次
        # 默认**关**。曾经默认开，实机症状：握拳时先卷四指（食指到 0），
        # 拇指随后收拢，index_floor 从 0 一路涨到 520，clamp 就把已经卷好的
        # 食指**顶了出去**——看上去像「握紧拇指会把食指弹开」。
        # 真正的过载保护是力控阈值（set_force，手内部固件执行），不是这里的钳位；
        # 而这组下限本身是捏合姿态量的，在握拳姿态下不成立。
        # 开着也不会再有那个症状了（见 clamp 里的单向规则），但没有实测支撑之前
        # 不默认开。要开：InspireHand6(couple_index_thumb=True)。
        self.couple_index_thumb = bool(couple_index_thumb)
        self._last_index_cmd: Optional[int] = None   # 单向规则用：上一次实际下发的食指值
        self._serial_factory = serial_factory
        self._ser = None
        self._io_lock = threading.Lock()

        self._target: Optional[list] = None
        self._measured: Optional[list] = None
        self._state_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self.io_errors = 0
        self.writes = 0
        self.clamped = 0

    # ── 连接 ────────────────────────────────────────────────────────────────
    def connect(self) -> "InspireHand6":
        if self._serial_factory is not None:
            self._ser = self._serial_factory()
        else:
            import serial
            self._ser = serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=8,
                                      parity="N", stopbits=1, timeout=self.timeout)
        angles = self.get_angles()      # 先证明链路通：CAN_H/CAN_L 接反读到的是静默，不是报错
        with self._state_lock:
            self._measured = list(angles)
        return self

    def close(self) -> None:
        self.stop_writer()
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    # ── 原始读写 ────────────────────────────────────────────────────────────
    def _transfer(self, tx: bytes) -> bytes:
        if self._ser is None:
            raise RuntimeError("未连接")
        last = None
        for _ in range(self.retries + 1):
            with self._io_lock:
                self._ser.reset_input_buffer()
                self._ser.write(tx)
                rx = self._ser.read(CanCodec.FRAME_LEN)
                # 转义后线上长度不定，先读 21 再按需补读
                while CanCodec.extract(rx) is None and len(rx) < CanCodec.MAX_FRAME_LEN + 4:
                    more = self._ser.read(4)
                    if not more:
                        break
                    rx += more
            frame = CanCodec.extract(rx)
            if frame is not None:
                return frame
            last = ProtocolError(f"未扫到合法帧 ({len(rx)} 字节): {rx.hex(' ') or '空'}")
            time.sleep(0.005)
        raise last

    def _read_chunk(self, addr: int, n: int) -> bytes:
        frame = self._transfer(CanCodec.build(CanCodec.CMD_READ, self.dev_id, addr, bytes([n]), 1))
        CanCodec.check(frame, addr, CanCodec.CMD_READ)
        return frame[6:6 + n]

    def _write_chunk(self, addr: int, payload: bytes) -> None:
        frame = self._transfer(
            CanCodec.build(CanCodec.CMD_WRITE, self.dev_id, addr, payload, len(payload)))
        CanCodec.check(frame, addr, CanCodec.CMD_WRITE)
        # 手册说写应答数据长度为 0，实测是 DLC=1、首字节=已写字节数。两种都算成功。
        if frame[14] == 0:
            return
        if frame[14] == 1 and frame[6] == len(payload):
            return
        raise ProtocolError(f"写应答异常 DLC={frame[14]} 首字节={frame[6]}")

    def read_bytes(self, addr: int, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            out += self._read_chunk(addr + len(out), min(CanCodec.MAX_PAYLOAD, n - len(out)))
        return bytes(out)

    def write_bytes(self, addr: int, payload: bytes) -> None:
        for off in range(0, len(payload), CanCodec.MAX_PAYLOAD):
            self._write_chunk(addr + off, payload[off:off + CanCodec.MAX_PAYLOAD])

    def read_i16(self, addr: int, n: int) -> list:
        return list(struct.unpack(f"<{n}h", self.read_bytes(addr, n * 2)))

    def read_u8(self, addr: int, n: int) -> list:
        return list(self.read_bytes(addr, n))

    def write_i16(self, addr: int, vals: Sequence[int]) -> None:
        self.write_bytes(addr, struct.pack(f"<{len(vals)}h", *[int(v) for v in vals]))

    # ── 语义接口 ────────────────────────────────────────────────────────────
    def clamp(self, six: Sequence[float]) -> list:
        """把 6 维目标钳进安全带。-1（保持不动）原样放行。

        食指的耦合下限（couple_index_thumb，默认关）是**单向**的：
        它只拦「食指正往拇指方向压下去」，绝不把已经卷好的食指往外推。

        之所以必须单向：下限是拇指弯曲量的函数，握拳时拇指后到，
        下限会在食指已经到 0 之后才涨起来。双向钳位在那一刻会把食指
        从 0 顶到 520 —— 实机上看到的就是「握紧拇指把食指弹开」。
        碰撞是「撞上去」才发生的，已经在下面的手指不会因为拇指靠近而变危险
        （拇指是压在卷好的四指外侧），所以拦下行、放上行才是对的。
        """
        out = []
        hit = False
        for i, v in enumerate(six):
            v = int(round(v))
            if v == ANGLE_HOLD:
                out.append(v)
                continue
            lo, hi = self.limits[i]
            c = max(lo, min(hi, v))
            if c != v:
                hit = True
            out.append(c)

        if self.couple_index_thumb and out[THUMB_BEND] != ANGLE_HOLD \
                and out[INDEX] != ANGLE_HOLD:
            prev = self._last_index_cmd
            floor = index_floor(out[THUMB_BEND])
            # 有效下限：还在下限之上时用真下限；已经在下限之下就用「当前位置」，
            # 于是最多是「不再往下」，永远不会被推回去。
            # 第一帧没有历史 —— 此时无从判断是压下去还是本来就在下面，
            # 不推，只记录；下一帧起规则就有依据了。
            if prev is None:
                eff = out[INDEX]
            else:
                eff = floor if prev >= floor else prev
            if out[INDEX] < eff:
                out[INDEX] = eff
                hit = True

        if out[INDEX] != ANGLE_HOLD:
            self._last_index_cmd = out[INDEX]

        if hit:
            self.clamped += 1
        return out

    def get_angles(self) -> list:
        return self.read_i16(ANGLE_ACT, NUM_DOF)

    def get_force(self) -> list:
        return self.read_i16(FORCE_ACT, NUM_DOF)

    def set_angles(self, six: Sequence[float]) -> None:
        """同步下发（会阻塞 ~15ms）。控制环里请用 post()。"""
        self.write_i16(ANGLE_SET, self.clamp(six))

    def set_force(self, f) -> None:
        vals = [int(f)] * NUM_DOF if isinstance(f, (int, float)) else [int(v) for v in f]
        for v in vals:
            if not 0 <= v <= 1000:
                raise ValueError(f"力控阈值 {v} 超出 0..1000")
        self.write_i16(FORCE_SET, vals)

    def set_speed(self, s) -> None:
        vals = [int(s)] * NUM_DOF if isinstance(s, (int, float)) else [int(v) for v in s]
        for v in vals:
            if not 0 <= v <= 1000:
                raise ValueError(f"速度 {v} 超出 0..1000")
        self.write_i16(SPEED_SET, vals)

    def get_status(self) -> list:
        return self.read_u8(STATUS_REG, NUM_DOF)

    def get_errors(self) -> list:
        return self.read_u8(ERROR_REG, NUM_DOF)

    def get_temperature(self) -> list:
        return self.read_u8(TEMP_REG, NUM_DOF)

    def liveness(self, samples: int = 3, gap: float = 0.7) -> tuple:
        """判断手的**内部控制循环**是不是还活着。返回 (ok, 说明)。

        为什么需要这个：手可能处在「通信层活着、控制层死了」的半死状态——
        寄存器读写全部正常（写 ANGLE_SET 能原样读回），但 ANGLE_ACT /
        FORCE_ACT / TEMP 全部冻结在一份旧快照上，手指一动不动。
        2026-09-15 实测遇到过：写 [900,800,700,...] 读回一模一样，
        而 ANGLE_ACT 六秒内逐位不变，连温度都不变。

        判据是**噪声**：真实采样量永远有抖动，温度和受力尤其。
        连续几次采样逐位完全相同 = 没在采样。
        （注意这和「用连续相同读数判手指停稳」不是一回事——那个会误判，
         因为静止的手指角度本来就会重复。这里看的是**多个物理量同时**
         一个 bit 都不变，而且包含温度这种必然漂移的量。）
        """
        snaps = []
        for i in range(max(2, samples)):
            if i:
                time.sleep(gap)
            try:
                snaps.append((tuple(self.get_angles()), tuple(self.get_force()),
                              tuple(self.get_temperature())))
            except Exception as exc:
                return False, f"读取失败: {type(exc).__name__}: {exc}"
        if all(s == snaps[0] for s in snaps[1:]):
            return False, (f"角度/受力/温度连续 {len(snaps)} 次采样逐位完全相同 —— "
                           "手的控制循环没在跑（通信层是活的，寄存器读写正常）")
        return True, "正常"

    def diagnose(self) -> str:
        errs, stats, temps = self.get_errors(), self.get_status(), self.get_temperature()
        return "\n".join(
            f"  {i} {n:<11} err={decode_error(errs[i]):<8} "
            f"status={STATUS_MEANING.get(stats[i], stats[i]):<12} temp={temps[i]}℃"
            for i, n in enumerate(DOF_NAMES))

    # ── 后台写线程 ──────────────────────────────────────────────────────────
    def start_writer(self) -> None:
        if self._worker is not None:
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._loop, name="atom-hand-io", daemon=True)
        self._worker.start()

    def stop_writer(self) -> None:
        if self._worker is None:
            return
        self._stop.set()
        self._wake.set()
        self._worker.join(timeout=2.0)
        self._worker = None

    def post(self, six: Sequence[float]) -> None:
        """非阻塞投递最新目标。旧的未发目标直接被覆盖（latest-wins）。"""
        with self._state_lock:
            self._target = self.clamp(six)
        self._wake.set()

    def measured(self) -> Optional[list]:
        with self._state_lock:
            return None if self._measured is None else list(self._measured)

    def _loop(self) -> None:
        n = 0
        while not self._stop.is_set():
            self._wake.wait(0.05)
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._state_lock:
                target, self._target = self._target, None
            try:
                if target is not None:
                    self.write_i16(ANGLE_SET, target)
                    self.writes += 1
                # 回读比写还贵（同样 2 帧 ≈ 14ms），每轮都读会把下发频率
                # 压到 ~35Hz。回读只服务于显示和诊断，降频即可，
                # 跟随的快慢取决于**写**的频率。
                n += 1
                if n % self.read_every:
                    continue
                m = self.get_angles()
            except Exception as exc:  # noqa: BLE001
                # 掉帧在这条总线上是常态，下一次写本来就会覆盖这一次。
                # 绝不让异常逃出线程——那会悄悄冻住整只手。
                self.io_errors += 1
                continue
            with self._state_lock:
                self._measured = m


# ── 离线自测用的假串口 ──────────────────────────────────────────────────────
class MockSerial:
    """按协议回帧的假串口，用来在没有硬件时验证编解码。"""

    def __init__(self):
        self.regs = bytearray(8192)
        struct.pack_into("<6h", self.regs, ANGLE_ACT, *[500] * 6)
        self._out = b""

    def reset_input_buffer(self):
        self._out = b""

    def close(self):
        pass

    def write(self, data):
        frame = CanCodec.extract(data)
        if frame is None:
            self._out = b""
            return len(data)
        can_id = int.from_bytes(frame[2:6], "little")
        addr = (can_id >> CanCodec.ADDR_SHIFT) & CanCodec.ADDR_MASK
        cmd = (can_id >> CanCodec.RW_SHIFT) & 0x7
        dlc = frame[14]
        if cmd == CanCodec.CMD_READ:
            n = frame[6]
            self._out = CanCodec.build(CanCodec.CMD_READ, 1, addr, bytes(self.regs[addr:addr + n]), n)
        else:
            self.regs[addr:addr + dlc] = frame[6:6 + dlc]
            # 写目标即刻反映到实际角度。注意 6 维要拆两帧，第二帧地址是
            # ANGLE_SET+8，所以这里要按**偏移**镜像，不能只认 addr == ANGLE_SET
            if ANGLE_SET <= addr < ANGLE_SET + NUM_DOF * 2:
                off = addr - ANGLE_SET
                self.regs[ANGLE_ACT + off:ANGLE_ACT + off + dlc] = frame[6:6 + dlc]
            self._out = CanCodec.build(CanCodec.CMD_WRITE, 1, addr, bytes([dlc]), 1)
        return len(data)

    def read(self, n):
        out, self._out = self._out[:n], self._out[n:]
        return out
