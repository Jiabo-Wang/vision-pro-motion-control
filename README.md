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

## 三条驱动路线，最后用的是第三条

这是这个项目的核心。同一套上层代码，臂可以走三条完全不同的路。前两条都跑通了，
但都不够跟手；**现在的答案是 servoJ**。

| | 本地 IK + 关节实时流 | 控制器 MoveL | **servoJ** |
|---|---|---|---|
| 入口 | `avp_arm_teleop.py --arm-backend rt` | `avp_movel_teleop.py` | **`avp_servoj_teleop.py`** |
| 控制模式 | `RtCommandMode` | `NrtCommandMode` | `RtCommandMode` + `setServoJoint` |
| 逆解在哪 | 本地 | **控制器** | 本地 |
| 下发什么 | 7 个关节角，1kHz | 笛卡尔路点，~4Hz | 7 个关节角，125Hz |
| 语义 | 流式覆盖式 | **提交任务式**，每个点必须走完 | **覆盖式**，新的盖掉旧的 |
| 谁插补 | 我们（Python） | 控制器 | **控制器**（8ms → 1ms） |
| 端到端延迟 | 低，但抖 | 240~500ms，手越快越差 | **约 100ms**，一半是主动加的 |
| 实际表现 | 抖动严重，跟手性不稳 | 一段一段走，臂在重放几秒前的动作 | **抖动、延迟、跟手性平衡最好** |

分界线是**队列**。MoveL 发进去的点排队等着被执行，手停了臂还要把队列走完，延迟等于
队列深度乘每点执行时间。servoJ 没有队列，发一个盖一个，控制器永远追最新那个 ——
同时又不用 Python 每毫秒准时发一次（那是 1kHz 关节流做不到的事）。代价是
servoJ 收关节角不收位姿，逆解和选解都得自己做。

五段同一动作的对照视频、每个参数的原理和调参顺序，在 [`v0.5_teleop.md`](v0.5_teleop.md)。
三条路各自在哪儿撞的墙，在 [`docs/AR5遥操三条路.md`](docs/AR5遥操三条路.md)。
更早的原理问答在 [`docs/问答/00-运动控制原理问答.md`](docs/问答/00-运动控制原理问答.md)。

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

### servoJ 的滞后就是三倍发送周期

不是玄学，是可以直接量出来的线性关系：

| 发送周期 | 实测滞后 |
|---|---|
| 8 ms | **24 ms** |
| 16 ms | 48 ms |
| 20 ms | 60 ms |
| 33 ms | 99 ms |

所以周期越小越跟手，下限卡在「Python 能不能准时发出来」。8ms 是这台机器上稳定不迟到的值
（实测 4234 次发送 0 次迟到）。

### 顺滑的开关是「主动落后」，不是加大滤波

逆解只在头显有新帧时更新（约 16.7ms 一次），servoJ 每 8ms 要一个点，中间靠插值。
关键在于**输出 50ms 之前那一刻的值**：因为落后了，输出点两侧都有真实数据，可以做
**内插**而不是外推。外推只能猜下一个目标在哪，目标一抖速度估计就跟着抖，样条过冲。

实测抖动指标 4637 → 2008，代价是 50ms 延迟。**低于 35ms 没意义** —— 右边取不到点，
会退化成保持不动，白搭延迟还更抖。

### 七轴的冗余必须显式管，否则臂会自己换姿势

同一个末端位姿，7 轴有无数组关节角。光挑「离上一帧最近」不够：实测一个位姿的
21 个候选分属**两个不同构型**，而构型之间没法连续过渡，中间要经过奇异。不筛的话
臂会在操作中途突然换个姿势。

做法是把构型压成三个比特（手腕正反、肘朝哪边、肩在前还是在后），接合时锚定，
之后只留同构型的解。另外把臂角锚在接合那一刻，搜索窗口钉在锚定值而不是上一帧的解 ——
只改这一处，臂角漂移从 30° 降到 0°。

### MoveL 慢的四个真因（都不是「控制器就这样」）

| 以为的原因 | 实际 |
|---|---|
| 每点有固定开销 | 执行时间 ≈ `2√(段长/加速度) + 74ms`，是**加速度受限**不是开销 |
| `speed` 就是速度 | 它同时决定**加速度档位**，分 5 档跳变：<100→10%、100~200→30%、200~500→50%、500~800→80%、>800→100% |
| `zone` 是转弯半径（毫米） | 是**百分比**。按 0.45×段长给会落进 10% 档，等于每个点都停一次 |
| 肘的位形配置一次就行 | 每次下发前都要刷新，否则臂会拧成很别扭的姿势 |

### 相机跑不到 30Hz 的两个独立原因

采数据时两路相机各只给 15~19Hz，查出来是两件互不相干的事：

- `exposure_dynamic_framerate` 开着 → 相机自己拿帧率换曝光，室内光照下卡在 **19Hz**，
  改曝光时间和像素格式都救不回来。
- OpenCV 的 `CAP_PROP_BUFFERSIZE=1` → **正好减半**，15.5Hz 对 29.3Hz。取图和还缓冲之间
  驱动没地方放下一帧，于是每隔一帧丢一帧。

判别方法：用 `v4l2-ctl --stream-mmap` 绕开 OpenCV 量一遍。raw v4l2 快而 OpenCV 慢
就是第二条，两个都慢就是第一条。两项都修完实测两路各 **29.6Hz**，录 90 帧零重复帧。

---

## 目录

Python 文件**刻意保持扁平**，和真机上的工作目录一一对应，方便双向同步。

### 主程序与模块

| 文件 | 作用 |
|---|---|
| [`avp_servoj_teleop.py`](avp_servoj_teleop.py) | **主程序**：servoJ 遥操，遥操就跑这一个 |
| [`ar5_poe_ik.py`](ar5_poe_ik.py) + [`paden_kahan.py`](paden_kahan.py) | 旋量法闭式解 |
| [`ar5_ik_select.py`](ar5_ik_select.py) | **选解层**：跳变门、构型筛选、代价排序 |
| [`ar5_sew.py`](ar5_sew.py) | 构型三比特判别、立体投影臂角 |
| [`hand_retarget.py`](hand_retarget.py) | 25 关节手骨架 → 手的 6 维；判手心手背 |
| [`inspire_hand6.py`](inspire_hand6.py) | RH56 驱动（CAN over USB 透传） |
| [`home_arm.py`](home_arm.py) | 单独把臂送回起始姿态 |
| [`vel_kin.py`](vel_kin.py) | 按文件路径加载外部运动学模块（绕开同名包遮蔽） |

对照用的另外两条路，留着做视频对比，不建议再用于遥操：

| 文件 | 作用 |
|---|---|
| [`avp_movel_teleop.py`](avp_movel_teleop.py) | 走控制器 MoveL，支持队列式和单点式两种发送 |
| [`avp_arm_teleop.py`](avp_arm_teleop.py) | 本地 IK + 1kHz 关节流（`--arm-backend rt`） |
| [`movel_stream.py`](movel_stream.py) | MoveL 路点流式下发 |
| [`ar5_ik.py`](ar5_ik.py) | `AR5OptIK`（优化式）/ `AR5IK`（DLS 迭代）/ `ToolFrameKinematics` |
| [`ar5_srs_ik.py`](ar5_srs_ik.py) | 从 DH 推的闭式解 |

### 数据采集

| 文件 | 作用 |
|---|---|
| [`episode_recorder.py`](episode_recorder.py) | 一条 episode 的内存缓冲 + **后台写盘**（主循环停 2 秒控制器就判丢包） |
| [`view_cameras.py`](view_cameras.py) | 开录前看相机、体检 |
| [`qc_episodes.py`](qc_episodes.py) | 采完一批的自动检查，十项 |
| [`replay_episode.py`](replay_episode.py) | 按编号回放，用眼睛确认 |
| [`manage_episodes.py`](manage_episodes.py) | 删条目、清孤儿 mp4、补编号 |
| [`to_lerobot_dex.py`](to_lerobot_dex.py) | 转 lerobot 数据集（**在 lerobot 环境里跑**） |

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
| [`probe_servoj.py`](probe_servoj.py) | 量 servoJ 的滞后和稳定性，不连头显 |
| [`probe_movel_speed.py`](probe_movel_speed.py) | 量 MoveL 一个点到底要多久、速度和加速度怎么挂钩 |
| [`probe_cameras.py`](probe_cameras.py) | 认相机：哪一路是俯视、哪一路是侧视 |
| [`viz_points.py`](viz_points.py) | 把生产/滤波/队列/执行各环节的点数画成动画 |

### 文档

| | |
|---|---|
| [`v0.5_teleop.md`](v0.5_teleop.md) | **主文档**：servoJ 全流程、参数原理、调参顺序、**五段对照视频** |
| [`docs/AR5遥操三条路.md`](docs/AR5遥操三条路.md) | 三条路各自的调试过程、矛盾点、在哪儿撞的墙 |
| [`docs/数据采集.md`](docs/数据采集.md) | 采数据全流程：相机检查、键盘约定、自动检查、编号策略 |
| [`docs/操作手册.md`](docs/操作手册.md) | 旧版操作手册，对应 `rt` / `movel` 两条路 |
| [`docs/实测数据.md`](docs/实测数据.md) | 所有真机实测数字汇总 |
| [`docs/问答/`](docs/问答/) | **整个项目 90 轮问答的完整记录**，按主题分 8 章 |

---

## 跑起来

> ⚠ **这个仓库不是自包含的。** 见下面「依赖」。

```bash
conda activate <你的环境>
cd vision-pro-motion-control

python avp_servoj_teleop.py <头显IP> --planner spline --lag-ms 50
```

回车挂离合，**右手保持捏合**臂才动，松开即停。按 `h` 停下回位，`q` 退出。
只有两个值要自己填：头显 IP（Tracking Streamer 界面上读，每次可能变）和 `--yaw`
（现场用 `probe_axes.py` 测一次，这台是 `-90`）。其余默认值就是实测调出来的。

参数怎么调、出问题查哪里，见 [`v0.5_teleop.md`](v0.5_teleop.md)。

采数据在这套遥操上面加 `--record`：

```bash
python view_cameras.py --check          # 先确认两路相机

python avp_servoj_teleop.py <头显IP> --planner spline --lag-ms 50 \
    --record --task "pick up the bolt" --num-episodes 50 --frame-cap 900

python qc_episodes.py                   # 采完自动查十项
python replay_episode.py --all --only-flagged
```

空格开录，`s` 存并标成功，`←` 存但不标成功，退格丢弃重录。整套流程见
[`docs/数据采集.md`](docs/数据采集.md)。

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
  疑似字节错位；两路相机都固定在台架上，**没有一路装在腕上**（`cam_front` 是侧视
  不是腕视）；控制器层的碰撞检测是关掉的，RSC 那道 17 Nm 的关节力限制 SDK 里
  没有任何接口能改，只能用珞石示教器软件动。
