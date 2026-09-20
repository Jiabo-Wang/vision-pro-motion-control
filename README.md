# Vision Pro → 珞石 AR5 + 因时 RH56 遥操

用 Apple Vision Pro 的手部追踪，实时驱动一台 **7 自由度珞石 AR5 机械臂** 和一只
**6 自由度因时 RH56 灵巧手**。左手控制臂的末端位姿和五指，右手捏合当离合（dead-man）。

这个仓库记录的不只是能跑的代码，还有**一整条调试路径** —— 包括好几条走到头才发现
是死路的方向。负面结果和正面结果在这里同等重要，因为它们都是在真机上量出来的。

---

## 硬件

| | |
|---|---|
| 机械臂 | 珞石 **AR5-5_0.8L-W4C5C11**，7 轴，S-R-S 构型 |
| 灵巧手 | 因时 **RH56**（带电容式触觉款），6 自由度，走 CAN over USB |
| 输入 | **Apple Vision Pro** + Tracking Streamer（头显做服务端，本机做客户端） |
| SDK | 珞石 **xCoreSDK** Python 绑定 v0.6.0 |

---

## 两条驱动路线

这是这个项目的核心。同一套上层代码，臂可以走两条完全不同的路：

| | `--arm-backend rt`（默认） | `--arm-backend movel` |
|---|---|---|
| 控制模式 | `RtCommandMode` | `NrtCommandMode` |
| 逆解在哪 | **本地**（旋量法 PoE 解析解，1.93ms） | **控制器** |
| 下发什么 | 7 个关节角，1kHz | 笛卡尔位姿路点，~4Hz |
| 语义 | **流式覆盖式**（新指令盖掉旧的） | **提交任务式**（每个点都必须走完） |
| 谁规划轨迹 | 我们 | 控制器 |
| 平滑靠什么 | 上游 One-Euro 滤波 + 速率限制 | 段长 + 转弯区 + 控制器 1kHz 插补 |
| 实测延迟 | — | **37ms 中位**（带 0.82mm 真实手抖） |
| 现场评价 | — | 「丝滑不少」 |

两条路的取舍、为什么最后是这么设计的，[`docs/问答/00-运动控制原理问答.md`](docs/问答/00-运动控制原理问答.md)
里从头讲了一遍。

---

## 几个值得单独拎出来的结论

### 抖动死区比加大滤波管用得多

One-Euro 滤波之后手的位姿还剩 **0.82mm** 残余抖动。这些抖动如果原样进 MoveL 队列，
臂就会去**逐个复刻它们** —— 而排队式是每个点都必须走完的，路径被切成毫米级碎片。

**短段是加速度受限的：峰值速度 ≈ √(加速度 × 段长)。** 实测抖动路径上臂只跑到
**25~33mm/s**，而指令是 250mm/s。

加一个 3mm 的**幅度死区**（不是加大滤波）：

| 死区 | 延迟中位 | p95 | 跟踪 RMS | 缓慢走 6mm 实到 |
|---|---|---|---|---|
| 0.5mm | 73ms | 114ms | 19.7mm | 3.79mm |
| 1.5mm | 61ms | 133ms | 19.8mm | — |
| **3.0mm** | **37ms** | **93ms** | **14.2mm** | **6.01mm** |

延迟砍掉一半，跟踪精度**反而变好**，连精细动作都更准。原因是死区作用在**幅度域**，
对超过阈值的真实运动**零滞后**；而滤波作用在频率域，全频段都拖慢。

### 「臂停了没」只能问控制器，不能看位置

队列走空臂就停，要补一次 `moveStart`。判据踩过两个坑，实机症状一模一样
**「动两下就不动了」**：

- 用「距上次下发多久」判 —— 臂完全能在持续下发的情况下把队列走光
- 改成看位置（连续 N 帧不动，≈60ms）仍太急 —— 臂物理上停了但控制器状态还没退，
  `moveStart` 被拒 `ec=-20`，14 秒里失败 14~20 次

改用 `operationState != moving` 之后降到 **0 次**。实测这个状态**比位置晚 87ms**
才退出 `moving`（位置停稳 270ms / 状态退出 357ms）—— 偏保守，正是需要的方向。
而且它只要 **0.093ms**，每帧查得起。

### 已经验死的路（别再试了）

| 方向 | 结论 | 证据 |
|---|---|---|
| **RT 笛卡尔模式** | 收指令但**不动** | 20mm 指令实到 0.4mm，**且不随指令变**（5mm 时也是 0.4mm）。1887 帧零异常，纯粹「收了什么都不做」 |
| **`FollowPosition_7`** | 用不了 | 依赖 `xmateModel`，而 `model.h` 里有硬编码白名单，AR5 不在其中 |
| **厂家 `calcIk`** | 能用但太慢 | 精度 0.0369°，但走 RPC **41ms**，50Hz 只有 20ms 预算 |
| **`moveStart` 异步化** | 完全无效 | xCoreSDK 的 Python 绑定**整个 C++ 调用期间不释放 GIL**：5 秒里工作线程读 26503 次、主线程只读到 1 次 |
| **靠 `wayPointIndex` 管队列深度** | 拿不到 | 它是**批内**下标，单条 append 时恒为 0 |

> 「模型库不支持 AR5」这个说法要小心：`libxMateModel.a` 里是 **80 个通用 Orocos KDL
> 目标文件**，没有任何机型专属数据。挡住 AR5 的是白名单，不是算不了。

### 本地逆解：旋量法解析解

从控制器读出真实 DH（[`read_dh.py`](read_dh.py) → [`ar5_dh.json`](ar5_dh.json)），
确认 AR5 是标准 S-R-S：

```
肩  j0,j1,j2 交于 (0, 0, 0.1745)    误差 0.23mm
肘  j3                               10.5mm 偏置
腕  j4,j5,j6 交于 (0, 0, 0.7845)    误差 0.32mm
```

于是可以用 **PoE（指数积）+ Paden-Kahan 子问题**写闭式解：

| 求解器 | 类型 | 耗时 | 残差 |
|---|---|---|---|
| `AR5PoeIK` | 解析（旋量法） | **1.93ms** | **0.0000mm** |
| `AR5OptIK` | 数值（L-BFGS-B + box 约束） | 2.65ms | 0.054mm |
| 厂家 `calcIk` | RPC | 41ms | — |

验证：PoE 正解 vs DH 正解一致到 **4.4e-16**；300/300 在真实臂角处恢复原解。

顺带一个反直觉的实测：理论上有 8 个解支，**这台臂平均只有 3.15 个够得着**，
其中**腕翻转 0.00% 可达**（J6/J7 只有 ±50°，翻 180° 转不过去）。

---

## 目录

Python 文件**刻意保持扁平**，和真机上的工作目录一一对应，方便双向同步。

### 主程序与模块

| 文件 | 作用 |
|---|---|
| [`avp_arm_teleop.py`](avp_arm_teleop.py) | **主程序**：头显 → 臂 + 手 |
| [`ar5_ik.py`](ar5_ik.py) | `AR5OptIK`（优化式，默认）/ `AR5IK`（DLS 迭代）/ `ToolFrameKinematics` |
| [`ar5_poe_ik.py`](ar5_poe_ik.py) + [`paden_kahan.py`](paden_kahan.py) | 旋量法解析解 |
| [`ar5_srs_ik.py`](ar5_srs_ik.py) | 从 DH 推的闭式解 |
| [`movel_stream.py`](movel_stream.py) | MoveL 路点流式下发（`movel` 后端） |
| [`hand_retarget.py`](hand_retarget.py) | 25 关节手骨架 → 手的 6 维；判手心手背 |
| [`inspire_hand6.py`](inspire_hand6.py) | RH56 驱动（CAN over USB 透传） |
| [`vel_kin.py`](vel_kin.py) | 按文件路径加载外部运动学模块（绕开同名包遮蔽） |

### 自检与标定

| 文件 | 作用 |
|---|---|
| [`test_changes.py`](test_changes.py) | **离线自检**，18 项断言，改完代码先跑 |
| [`probe_axes.py`](probe_axes.py) | **定 `--yaw` / `--mirror` + 验手心手背**，不碰臂 |
| [`test_fingers.py`](test_fingers.py) | 逐根手指自检 |
| [`bench_ik.py`](bench_ik.py) / [`check_fk.py`](check_fk.py) | 求解器对比基准 / 正解校验 |
| [`read_dh.py`](read_dh.py) | 从控制器读真实 DH |
| [`mock_avp.py`](mock_avp.py) | 假头显，没设备时调参用 |

### 探路脚本（都是独立的，不依赖本项目模块）

结论写在各自文件头里，**别重复踩**：

| 文件 | 回答了什么 |
|---|---|
| [`try_cartesian_rt.py`](try_cartesian_rt.py) | RT 笛卡尔能不能用 → **不能** |
| [`diag_cartesian_rt.py`](diag_cartesian_rt.py) | 卡在哪一环 → `pos_c` 从不更新，指令没进规划器 |
| [`try_movel_stream.py`](try_movel_stream.py) | MoveL 排队式能不能撑遥操 → **能** |
| [`diag_movel_state.py`](diag_movel_state.py) | `operationState` 靠不靠得住、`moveStart` 各返回码什么意思 |
| [`try_async_start.py`](try_async_start.py) | `moveStart` 异步化有没有用 → **没用**，绑定不放 GIL |

### 文档

| | |
|---|---|
| [`docs/操作手册.md`](docs/操作手册.md) | **怎么跑**：⓪→⑤ 分阶段清单、坐标系、参数表、故障对照表 |
| [`docs/实测数据.md`](docs/实测数据.md) | 所有真机实测数字汇总 |
| [`docs/问答/`](docs/问答/) | **整个项目 90 轮问答的完整记录**，按主题分 8 章 |

---

## 跑起来

> ⚠ **这个仓库不是自包含的。** 见下面「依赖」。

```bash
conda activate <你的环境>
cd vision-pro-motion-control

python avp_arm_teleop.py <头显IP> --no-cameras --arm-backend movel \
    --yaw -90 --payload 1.0 --payload-com-z 0.08
```

右手捏合接合，再捏一次松开。只有三个值要自己填：头显 IP、`--yaw`（现场用
`probe_axes.py` 测一次）、`--payload`（手的质量，默认 0 会让控制器把自重当碰撞）。
其余默认值就是实测调出来的最优。

完整流程和每一步该看什么，见 [`docs/操作手册.md`](docs/操作手册.md)。

### 依赖

- **珞石 xCoreSDK Python 绑定**（厂家提供，不在本仓库）。脚本里的 SDK 路径是硬编码的
  绝对路径，需要改成你自己的。
- **`avp_stream`**（Vision Pro Tracking Streamer 的 Python 客户端）。
- **一个外部运动学模块**：[`vel_kin.py`](vel_kin.py) 里的 `AR5_ROOT` 指向本仓库之外的
  另一个项目。那个项目不属于这里，**克隆本仓库后需要自行提供或改写这一层**。
- numpy / scipy / pyserial，见 [`requirements.txt`](requirements.txt)。

### 里面的 IP 和路径

代码里写死了实验室的内网地址（臂 `192.168.2.160`、本机 `192.168.2.222`）和
`/home/.../` 开头的绝对路径。都需要按你的环境改。

---

## 说明

- 所有性能数字都是在**真机上量的**，不是仿真。方法和原始输出在
  [`docs/实测数据.md`](docs/实测数据.md) 和各探路脚本的文件头注释里。
- 代码注释是中文，而且刻意写得长 —— 很多注释记的是「为什么不能那样写」，
  那些是踩过坑才知道的。
- **还没做完的**：TCP 和负载仍是估值；触觉寄存器读出来的值超出文档量程，
  疑似字节错位；真头显的端到端延迟没有独立测过（本文数字用的是合成手抖）。
