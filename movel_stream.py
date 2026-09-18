#!/usr/bin/env python3
"""MoveL 多路点流式下发 —— 本地只算绝对目标，**逆解交给控制器**。

设计取向由现场定：非阻塞、延迟控在 ~100ms、控制器做绝对位姿推算。

── 延迟怎么控住 ────────────────────────────────────────────────────────
不能用队列深度：`wayPointIndex` 是**批内**下标（实测单条 append 时恒为 0，
批量 19 点时才会 0→18 地走），跨批次拿不到进度，`cmdID` 也不回显。

改用**前导距离**：

    延迟 ≈ (最后一个已下发目标 − 当前实测位置) / 速度

这个每帧都算得出来（`cartPosture` 实测 0.177ms）。想要 100ms 延迟、
速度 200mm/s，就把前导距离压在 20mm 以内：超了就**跳过这一帧**
（目标是"最新值"，不是必须执行的轨迹段），没超就 append。

⚠ 前导距离是**直线距离**，而队列里排的是一条**折线**。目标带抖动时折线远比
直线长，臂要挨个走完 —— 所以噪声大的时候，`lead/speed` 会**低估**真实延迟。
别只盯这个数，`min_step` 死区（见下）往往是更有效的那个旋钮。

── 队列跑空了怎么重启 ────────────────────────────────────────────────
实测：`moveStart` 单次往返 **42.8ms**，首次运动延迟 **127ms**。
队列一旦执行完，臂停下，恢复就要再吃一次。手停住时**不清空队列**，
而是停止 append 让它自然走完；手再动时若判定已停，补一次 moveStart。

**「已停」必须问控制器（`operationState != moving`），不能看位置。**
看位置（连续 N 帧不动）踩过两个坑，实机症状都是「动两下就不动了」：
  · 判据本身错 —— 原来用「距上次 append 多久」，可臂完全能在我持续 append
    的情况下把队列走光停下，那时 need_start 恒假，新点永远没人启动。
  · 改成看位置后仍太急 —— 臂物理上停了但控制器运动状态还没退，
    `moveStart` 被拒 `ec=-20 机器人运动中`，14 秒内失败 14~20 次。
换成 operationState 后降到 0~1 次（0.093ms/次，每帧查得起）。

── moveStart 的返回码（实测，见 diag_movel_state.py）───────────────────
    ec=0     成功
    ec=-20   「机器人运动中」—— **良性**。Q3 实测：运动中 append 的点会被
             控制器**自动接上执行**（指令 6mm 实到 6.0mm），不用补 start。
    ec=768   「没有可执行的运动指令」= 队列已空 —— 也是良性，没东西要跑。
             我一度在补救路径上空放这个码，14 秒报 28 次「失败」，吓人但无害。
只有这两个之外的码才是真失败。

── operationState 靠不靠得住（Q1 实测）────────────────────────────────
一段 6mm 的 MoveL：位置停稳 270ms，状态退出 moving 357ms ——
**状态比位置晚 87ms**。方向是对的：它绝不会在臂还动的时候说 idle，
只会保守一点。状态序列干净：moving → idle。

⚠ `moveStart` 是同步往返，会把调用方阻塞 42.8ms，而且恰好发生在运动恢复的
瞬间。50Hz 下实测 `update()` p95 43ms、最大 87ms。**放到工作线程上没用**：
xCoreSDK 的 Python 绑定整个 C++ 调用期间不放 GIL（实测 5 秒里工作线程读
26503 次、主线程 1 次），换线程照样堵死主循环。见 `try_async_start.py`。

── 速度和死区怎么定 ──────────────────────────────────────────────────
**干净**目标下走走停停（动 1.2s 停 0.8s）：
    150mm/s → 66ms，跳帧 52%     250mm/s → 41ms，30%     400mm/s → 41ms，21%
但这偏乐观。换成带 0.82mm 残余手抖的真实输入，250mm/s 是 **73ms**。
此时更有效的旋钮是死区（速度固定 250mm/s）：
    死区 0.5mm → 73ms / p95 114ms / 跟踪RMS 19.7mm
    死区 1.5mm → 61ms / p95 133ms / 跟踪RMS 19.8mm
    死区 3.0mm → 37ms / p95  93ms / 跟踪RMS 14.2mm   ← 默认，延迟精度双赢

为什么死区这么有效：**短段是加速度受限的，臂根本跑不到指令速度**
（实测抖动路径上臂只有 25~33mm/s，指令是 250）。死区把段拉长，臂才提得起速。

死区**不会**吃掉精细动作，反而更准。8 秒缓慢走 6mm（0.75mm/s，比死区本身还慢）：
    死区 0.5mm → 实到 3.79mm（下发 214 条，臂在追噪声，跟不上）
    死区 3.0mm → 实到 6.01mm（下发 2 条，精准到位）
死区是对**上一个已下发点**的增量阈值，慢速运动会累积过阈，一个都不丢。

── 各调用实测耗时（AR5-5_0.8L，静止态）────────────────────────────────
    moveAppend(1条)                0.03 ms   ← 只塞队列，不阻塞
    moveAppend(20条批量)           0.44 ms
    moveStart                     42.78 ms   ← 有往返，但非阻塞（返回时臂未动）
    cartPosture(flangeInBase)      0.177 ms
    queryEventInfo                 0.002 ms
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np


class MoveLStreamer:
    """把绝对目标位姿流式喂给控制器。调用方每帧给一个目标，其余这里管。

    用法：
        s = MoveLStreamer(robot, sdk, speed=200, lead_ms=100)
        s.configure()          # 设速度/转弯区/取最近解
        s.start()
        while teleop:
            s.update(target_xyz, target_rpy)   # 50Hz，绝对位姿
        s.stop()
    """

    def __init__(self, robot, sdk, *, speed: float = 200.0, zone: float = 3.0,
                 lead_ms: float = 100.0, min_step_mm: float = 0.5,
                 cache: int = 300):
        self.robot, self.sdk = robot, sdk
        self.speed = float(speed)              # mm/s
        self.zone = float(zone)                # mm，转弯区，相邻段混合
        # 前导距离上限 = 速度 × 延迟预算。
        self.max_lead = self.speed * lead_ms / 1000.0 / 1000.0   # 米
        # 抖动死区。**延迟的第二个旋钮，而且往往比速度更管用**：
        # One-Euro 之后仍有 ~0.82mm 残余抖动，死区比它小的话抖动会被原样塞进
        # 队列，臂就去追噪声 —— 队列里的**路径长度**因此暴涨，而排队式是每个点
        # 都必须走完的，延迟就跟着涨。死区对真实运动零滞后（过阈值原样通过），
        # 比继续加大滤波划算。
        self.min_step = min_step_mm / 1000.0
        self.cache = int(cache)
        self._last_target: Optional[np.ndarray] = None
        self._last_rpy: Optional[list] = None
        self._running = False
        # 有没有「已经 append 但可能还没被启动」的点。
        # moveStart 返回 ec=-20（机器人运动中）时**不能**当成已经启动：控制器
        # 可能正在收尾上一批，随即判定结束停下，这个点就永远搁浅了。
        # 连续遥操时下一帧 append 会顺带把它启动（自愈），但**停手的瞬间**
        # 没有下一帧 —— 实测缓慢走 6mm 只走到 2.74mm 就是这么丢的。
        self._pending = False
        self.stats = {"appended": 0, "skipped_lead": 0, "skipped_tiny": 0,
                      "restarts": 0, "already_moving": 0, "queue_empty": 0,
                      "start_failed": 0, "errors": 0}

    # ── 配置 ────────────────────────────────────────────────────────────
    def wait_until_stopped(self, timeout: float = 8.0, log=print) -> bool:
        """等臂**真的停稳**。

        `_stop_rt_thread()` 只是让发送器刹车，臂还在减速。这时候切控制模式/上电
        一律被拒：`ec=-20 机器人运动中`，而且是**静默失败**（ec 在字典里，不抛异常），
        后面 MoveL 全程无效却看不出原因。实机上就栽在这里。
        判据用位置连续不变，比问状态可靠。
        """
        t0 = time.perf_counter()
        last = self.tcp()
        still = 0
        while time.perf_counter() - t0 < timeout:
            time.sleep(0.05)
            cur = self.tcp()
            if float(np.linalg.norm(cur - last)) < 2e-5:
                still += 1
                if still >= 5:
                    return True
            else:
                still = 0
            last = cur
        log(f"  [MoveL] ⚠ 等了 {timeout:.0f}s 臂还在动，继续会被拒(ec=-20)")
        return False

    def configure(self, log=print):
        """排队式运动要 Nrt 模式；RT 模式下 Move* 会被拒（ec=262 运动控制模式错误）。

        调用前臂必须**已经停稳**，否则每一步都会 ec=-20 且静默失败。
        """
        sdk = self.sdk
        if not self.wait_until_stopped(log=log):
            log("  [MoveL] 臂没停稳就配置，下面多半会失败")
        steps = [
            ("NrtCommandMode", self.robot.setMotionControlMode,
             sdk.MotionControlMode.NrtCommandMode),
            ("automatic", self.robot.setOperateMode, sdk.OperateMode.automatic),
            ("上电", self.robot.setPowerState, True),
            # 逆解取**离当前轴角最近**的解 —— 分支选择交给控制器，正是遥操要的
            ("取最近解", self.robot.setDefaultConfOpt, False),
            ("缓冲路点", self.robot.setMaxCacheSize, self.cache),
            ("默认速度", self.robot.setDefaultSpeed, self.speed),
            ("转弯区", self.robot.setDefaultZone, self.zone),
        ]
        bad = []
        for name, fn, arg in steps:
            ec = {}
            try:
                fn(arg, ec)
                if ec.get("ec", 0):
                    bad.append(f"{name}(ec={ec.get('ec')} {ec.get('message','')})")
            except Exception as exc:                    # noqa: BLE001
                bad.append(f"{name}({type(exc).__name__}: {exc})")
        if bad:
            log(f"  [MoveL] ✗ 配置失败 {len(bad)}/{len(steps)} 项: {'; '.join(bad)}")
            log("        这些失败是**静默**的（ec 在字典里不抛异常），"
                "不报出来的话 MoveL 全程无效却看不出原因")
            return False
        log(f"  [MoveL] ✓ 配置完成（{len(steps)} 项）")
        return True

    # ── 状态 ────────────────────────────────────────────────────────────
    def is_idle(self) -> bool:
        """控制器说自己不在运动中。比看位置可靠：位置停了不代表状态退了。

        实测 operationState 只要 0.093ms（p95 0.54ms），每帧查得起。
        """
        try:
            ec = {}
            st = self.robot.operationState(ec)
            return st != self.sdk.OperationState.moving
        except Exception:                               # noqa: BLE001
            return False

    def tcp(self) -> np.ndarray:
        ec = {}
        cp = self.robot.cartPosture(self.sdk.CoordinateType.flangeInBase, ec)
        return np.asarray(cp.trans, dtype=float)

    def _cmd(self, trans, rpy):
        t = self.sdk.CartesianPosition()
        t.trans = [float(v) for v in trans]
        t.rpy = [float(v) for v in rpy]
        # 冗余臂：带上臂角和构型，否则控制器可能解不出来
        if self._conf is not None:
            t.elbow = self._elbow
            t.hasElbow = True
            t.confData = list(self._conf)
        c = self.sdk.MoveLCommand(t)
        c.speed = self.speed
        c.zone = self.zone
        return c

    # ── 生命周期 ────────────────────────────────────────────────────────
    def start(self, log=print):
        ec = {}
        cp = self.robot.cartPosture(self.sdk.CoordinateType.flangeInBase, ec)
        self._elbow = cp.elbow
        self._conf = list(cp.confData)
        self._last_rpy = list(cp.rpy)
        self._last_target = np.asarray(cp.trans, dtype=float)
        ec = {}
        self.robot.moveReset(ec)
        self._running = True
        self._pending = False
        log(f"  [MoveL] 起点 {np.round(self._last_target*1000,2).tolist()} mm  "
            f"速度 {self.speed:.0f}mm/s  转弯区 {self.zone:.0f}mm  "
            f"前导上限 {self.max_lead*1000:.0f}mm(≈{self.max_lead/self.speed*1e6:.0f}ms)")

    def update(self, trans, rpy=None) -> str:
        """喂一个**绝对**目标位姿。返回这一帧做了什么。"""
        if not self._running:
            return "stopped"
        trans = np.asarray(trans, dtype=float)
        rpy = self._last_rpy if rpy is None else list(rpy)

        # 目标几乎没动 —— 不下发。让队列自然走完，别塞退化段
        # （控制器对过近的点会在 Remark 里告警）
        if np.linalg.norm(trans - self._last_target) < self.min_step:
            self.stats["skipped_tiny"] += 1
            # 兜底：真正启动失败（不是 -20 / 768 那两个良性码）时留下的欠账，
            # 手停住后没有新的 append 能顺带救它，只能在这里补。
            # 正常情况下 _pending 早被清了，这条路几乎不开火。
            if self._pending and self.is_idle():
                self._try_start()
            return "tiny"

        cur = self.tcp()
        # 「臂停了没」问控制器，不看位置。
        # 用位置判（连续 N 帧不变）太急：臂物理上停了，控制器的运动状态还没退，
        # 这时 moveStart 会被拒 `ec=-20 机器人运动中`。实测失败 14~20 次/14秒。
        # operationState 是权威判据，实测只要 0.093ms，每帧查得起。
        stopped = self.is_idle()

        # 前导距离超预算 → 跳过。目标是最新值，不是必须执行的轨迹。
        # 但**臂已经停了就不能跳** —— 跳了就没人再启动它，永远停在那。
        lead = float(np.linalg.norm(self._last_target - cur))
        if lead > self.max_lead and not stopped:
            self.stats["skipped_lead"] += 1
            return "lead"

        ec = {}
        try:
            self.robot.moveAppend(self._cmd(trans, rpy),
                                  self.sdk.PyString(f"m{self.stats['appended']}"), ec)
            if ec.get("ec", 0):
                self.stats["errors"] += 1
                return "err"
        except Exception:                               # noqa: BLE001
            self.stats["errors"] += 1
            return "err"
        self.stats["appended"] += 1
        self._last_target = trans
        self._last_rpy = rpy
        self._pending = True

        if stopped:
            return "append+" + self._try_start()
        return "append"

    def _try_start(self) -> str:
        """启动队列。只有**确认成功**才清 `_pending`。"""
        ec = {}
        try:
            self.robot.moveStart(ec)
            # ⚠ 必须看 ec。moveStart 的失败是**返回错误码**不是抛异常，
            # 只 try/except 会把失败当成功，队列再也没被启动 —— 臂就停那了。
            code = ec.get("ec", 0)
        except Exception as exc:                        # noqa: BLE001
            self.stats["start_failed"] += 1
            return f"start异常({type(exc).__name__})"
        if code == -20:
            # 「机器人运动中」——**良性**，实测确认（diag_movel_state.py 的 Q3）：
            # 运动中 append 的点会被控制器**自动接上执行**，Y 轴指令 6mm 实到 6.0mm。
            # 所以欠账已经清了，不用补 start。
            self.stats["already_moving"] += 1
            self._pending = False
            return "已在动"
        if code == 768:
            # 「没有可执行的运动指令」= 队列已空。也是良性：没东西要跑。
            # 我一度在补救路径上空放这个码，14 秒里报 28 次「失败」，
            # 吓人但无害 —— 别再把它记成失败。
            self.stats["queue_empty"] += 1
            self._pending = False
            return "队列空"
        if code:
            self.stats["start_failed"] += 1
            return f"start失败({code})"
        self.stats["restarts"] += 1
        self._pending = False
        return "start"

    def stop(self, go_home: Optional[np.ndarray] = None, log=print):
        self._running = False
        ec = {}
        try:
            self.robot.moveReset(ec)
        except Exception:                               # noqa: BLE001
            pass
        if go_home is not None:
            ec = {}
            try:
                self.robot.moveAppend(self._cmd(go_home, self._last_rpy),
                                      self.sdk.PyString("home"), ec)
                ec = {}
                self.robot.moveStart(ec)
            except Exception as exc:                    # noqa: BLE001
                log(f"  [MoveL] 回位失败: {exc}")

    def report(self) -> str:
        s = self.stats
        tot = s["appended"] + s["skipped_lead"] + s["skipped_tiny"]
        out = (f"MoveL 流式: 下发 {s['appended']}  "
               f"跳过(超前导) {s['skipped_lead']}  跳过(没动) {s['skipped_tiny']}  "
               f"重启 {s['restarts']}  已在动 {s['already_moving']}  "
               f"队列空 {s['queue_empty']}  错误 {s['errors']}  (共 {tot} 帧)")
        if s["start_failed"]:
            out += (f"\n  ⚠ **moveStart 失败 {s['start_failed']} 次**（不含良性的"
                    "「已在动」）—— 队列没被启动，臂会停住不动")
        return out
