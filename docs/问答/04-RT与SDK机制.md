# RT / Nrt 与厂家 SDK 机制

实时模式到底是什么、RT 笛卡尔为什么收指令不动、模型库白名单、FollowPosition_7 为什么用不了、笛卡尔增量 API 在哪。

> **这是一份按时间顺序的真实调试记录，不是一份已经理顺的教程。**
>
> 里面有相当一部分中间结论**后来被推翻了** —— 那正是这份记录的价值所在，
> 它记着每个结论是怎么一步步试出来的、错在哪、怎么发现错的。几个显眼的例子：
>
> | 中间结论 | 后来发现 |
> |---|---|
> | 「模型库不支持 AR5」 | 错。`libxMateModel.a` 是 80 个通用 KDL 目标文件，真正的限制是 `model.h` 里的硬编码白名单 |
> | 「`moveAppend` 要 42ms」 | 错。那是 `checkPath` 的数，`moveAppend` 只要 **0.03ms**。这个混淆一度把整条排队路径判死 |
> | 「`ec=-20` 会让路点搁浅」 | 错。实测运动中 append 的点会被控制器自动接上执行 |
> | 「延迟 41ms」 | 偏乐观。那是拿干净目标测的；带真实手抖是 73ms，加死区后才回到 37ms |
>
> **当前的正确结论在 [`00-运动控制原理问答`](00-运动控制原理问答.md) 和
> 仓库根目录的 [`README`](../../README.md) 里**，以那两处为准。


---


## 问 1

> 我现在怀疑传感器捕捉有问题，我现在人手是增量变化，机械臂也是增量变化吗


**对应关系是绝对的，但传输是增量的。** 具体说：

```python
p_target = ref.p_tool + R @ (here − ref.p_hand) × scale   # ← 绝对：手在哪，末端就在哪
delta    = (p_target − cur6) × gain                       # ← 转成增量
env.step(delta)  →  new_target = _base_pose() + delta     # ← 积分回去
```

所以「手移动 10cm，末端就在基准外 10cm」是绝对的，不累积漂移。但这个绝对性**完全取决于 `ref`**，而 `ref` 一旦被重置，对应关系就整体平移一次——**重置得够频繁，绝对映射就退化成增量映射了**。

而我上一轮很可能把重置弄得过于频繁了：

模拟里重置率很低（0.05 次/秒），**所以我不能说这就是原因**——我上一段的怀疑没被数据支持。

回到你的问题，准确答案是：

| | |
|---|---|
| **对应关系** | **绝对**——手在基准外 10cm，末端就在基准外 10cm，不累积漂移 |
| **传输方式** | **增量**——算出绝对目标后转成 delta 交给 `env.step()` 积分 |

但绝对性只在 `ref` 不变时成立，代码里有 **5 处**会清掉 `ref`（松手、打滑重锚、限位自救……），每清一次对应关系就整体平移一次。

你怀疑传感器——这个可以**不接机械臂**直接验。做个只连头显的记录器：

工具本身跑通了（上面那些「漂移/标度不对」是 mock 在做正弦运动的产物，不是真数据——mock 手腕一直在动，所以「静止段」量到 96mm 漂移是预期的）。

回答你的问题：

## 人手绝对，机械臂也是绝对的 —— 但只在 `ref` 不变时

```python
p_target = ref.p_tool + R @ (here − ref.p_hand) × scale   # 绝对对应
delta    = (p_target − cur6) × gain                       # 转成增量
env.step(delta) → new_target = _base_pose() + delta       # 积分回去
```

**对应关系是绝对的**（手在基准外 10cm，末端就在基准外 10cm，不累积漂移），**传输是增量的**。

但代码里有 **5 处**会清掉 `ref`，每清一次对应关系就整体平移一次。重置得够频繁，绝对映射就退化成你感觉到的那种"增量"行为了。我本来怀疑是我上一轮把「跟不上」判据（位置>10mm **或姿态>15°**）接进打滑逻辑导致重置过频——但**模拟不支持这个猜测**，温和轨迹上只有 0.05 次/秒。所以这条存疑，得看真机数据。

## 你的怀疑可以直接验，不用接臂

```bash
python probe_hand_data.py 192.168.0.246
```

按提示做两个动作（举着不动 10 秒；平移 30cm 再停），它量三项：

1. **静止漂移 / 噪声** — 毫米级抖动 = 臂抖就是数据抖，调臂没用
2. **帧到达间隔** — 忽大忽小 = 网络在卡，遥操拿到陈旧帧，滤波治不了
3. **位移标度比** — 你走 30cm 它报多少？**这条直接回答你的问题**：如果标度不对，人手增量和机械臂增量从源头就对不上，`--scale` 也补不回来

第 2 项我有实际理由怀疑：这台机器 ping 头显是 **5~407ms，mdev 130ms**，局域网里这个抖动很不正常。

三项都正常 → 数据没问题，回头查臂那一侧；有一项不对 → 就锁定了，不用再在臂上瞎调。

跑之前记得头显戴上、Tracking Streamer 点 Start（上次就是没启动）。


---


## 问 2

> 把坐标系改到末端才是我想要的效果；
> 你的IK反解是你自己解的，把state输出给机械臂去执行，还是你把增量给机械臂，机械臂自己用控制器去做IK反解


这条注释是答案的关键：

> **The xMate model library does not support AR5**: `robot.model()` raises "xMate模型库暂不支持的机型". So there is **no SDK forward/inverse kinematics or Jacobian for this arm**.

查一下控制器侧的 IK 到底有没有这条路：

补丁把原来的帮助文本截断了，留下了孤儿行。修：

末端系已设为默认（`--delta-frame tool`），回退用 `--delta-frame base`。

## 你的问题：**我自己解，把关节角发给机械臂**

控制器不做任何 IK。完整链路：

```
遥操算出绝对目标 → delta（笛卡尔）
  → env.step(delta)  →  compose_pose  →  笛卡尔位姿
  → servo_to_pose_rt(pose)
       → self.kinematics.ik(T, seed)          ← **我的 Python 逆解**（AR5OptIK）
       → servo_to_joints(q)                    ← 四道闸：有限值/宽度/限位/跳变
  → 1kHz 线程: sendCommand(JointPosition(q))   ← **发出去的是 7 个关节角**
```

两处硬证据：

```python
rokae_arm.py:371  self._rt.startMove(RtControllerMode.jointPosition)   # 关节位置模式
rokae_arm.py:566  self._rt.sendCommand(JointPosition(q_cmd.tolist()))  # 发关节角
```

**为什么是这样**——驱动头部的注释记了原因，不是随意选的：

> **The xMate model library does not support AR5**：`robot.model()` 抛「xMate模型库暂不支持的机型」。所以这台臂**没有 SDK 侧的正逆运动学和雅可比**。

所以本地这套 DH 运动学是自己搭的（也正因如此我才去读了控制器的标定 DH 表）。

## 但有一条路可能没被试过

我刚查了 SDK，`RtControllerMode` 里**确实有 `cartesianPosition`**：

```
['cartesianImpedance', 'cartesianPosition', 'jointImpedance', 'jointPosition', 'torque']
```

这意味着理论上可以**直接发笛卡尔位姿，让控制器在 1kHz 上自己做 IK**。注意那条注释说的是「**SDK** 模型库不支持 AR5」——那是客户端侧的库；控制器自己肯定知道自己的运动学（它能做 MoveL）。这两件事不是一回事。

如果这条路通，好处很实在：IK 在 1kHz 上做而不是 50Hz、奇异和限位由厂商的控制器原生处理、我这边所有关于步长/跳变闸的补丁都不需要了。

**但我没试过，不能保证它对 AR5 可用。** 要试就得给真机发笛卡尔 RT 指令，属于会动臂的操作。你先把末端系这版跑一下，如果手感对了，要不要再花时间去验 `cartesianPosition` 这条路，你定。

```bash
python avp_arm_teleop.py 192.168.0.246 --no-cameras --yaw -90 --lock-j5 0.0
```

（`--delta-frame tool` 现在是默认，不用写。头显记得先点 Start。）


---


## 问 3

> 你能不能找到厂家SDK里有笛卡尔坐标的增量API


SDK 类列表里有 **`CartesianPositionOffset`** 和 **`CartesianPositionOffsetType`** —— 名字就是笛卡尔增量。全面查一遍：

`relTool` = **相对工具坐标系偏移** —— 正是你要的末端系增量，`offs` 是基座系。控制器原生支持两种。现在找哪些接口吃这个类型：

这些全是**排队式** `Move*` 指令的参数（`MoveLCommand.offset` 等），不是实时路径。而排队路径是 42ms 往返 + 持 GIL，50Hz 遥操用不了。

关键是实时路径能不能吃笛卡尔。查 `sendCommand` 的重载：

RT 控制器不是顶层符号（由 `getRtMotionController()` 返回，绑定类是 `PyRTmotioncontrol7`）。从 .so 里直接抓它的签名：

`setControlLoopCar`（笛卡尔控制环）存在，还有 `cartPos/cartVel/cartAcc`。查 RT 支持的字段和 `sendCommand` 能吃什么：

`RtSupportedFields` 是**读**的字段（`_m` = measured），含 `tcpPose_m`（末端位姿）和 `elbow_m`（臂角），可以 1kHz 读。写的一侧要看 `sendCommand` 的重载。从二进制里找它的中文文档：

示例里发现 **`robot.model().calcIk(cart_pos, toolset, ec)`** —— SDK 自带逆解。但驱动注释记过 AR5 用不了（`robot.model()` 抛「xMate模型库暂不支持的机型」）。继续找实时路径的笛卡尔写入：

找到了，但结论有个关键限制。

## 有，而且原生支持你要的「末端系增量」

```python
CartesianPositionOffsetType:
    none     无偏移
    offs     相对**工件/基座**坐标系偏移     ← 对应我的 --delta-frame base
    relTool  相对**工具**坐标系偏移          ← 对应 --delta-frame tool
```

用法（本地构造验证过）：

```python
fr = sdk.Frame()                              # trans[3] + rpy[3]
fr.trans = [0.01, 0, 0]                       # 末端系 +X 走 10mm
off = sdk.CartesianPositionOffset(sdk.CartesianPositionOffsetType.relTool, fr)
cmd = sdk.MoveLCommand(); cmd.offset = off
```

厂家把「基座系 vs 工具系」这个区分做进了 API —— 说明这是个标准需求，我们上一轮手工实现的两种模式，控制器本来就有。

## 但它**只在排队式指令上**

`CartesianPositionOffset` 是 `MoveLCommand` / `MoveJCommand` / `MoveCCommand` / `MoveSPCommand` 的字段，**不在实时路径上**。而排队路径是 `moveAppend` + `moveStart`，vel_ar5 实测 **42ms 往返且持 GIL**，50Hz 遥操用不了——一帧 20ms 预算，光一次调用就超一倍。

## 实时路径的情况（顺带查清了）

| 能力 | 有没有 |
|---|---|
| `RtControllerMode.cartesianPosition` | ✅ 存在 |
| `setControlLoopCar`（笛卡尔控制环） | ✅ 存在 |
| `cartPos / cartVel / cartAcc` 字段 | ✅ 存在 |
| 读：`tcpPose_m`（末端位姿）、`elbow_m`（臂角） | ✅ 1kHz 可读 |
| **笛卡尔增量** | ❌ 实时路径只有绝对位姿，没有 offset |

还有一个：`robot.model().calcIk(cart_pos, toolset, ec)` —— **SDK 自带逆解**。但驱动注释记过 AR5 用不了（`robot.model()` 抛「xMate模型库暂不支持的机型」）。这条**我没实测过**，只是引用了别人的记录；值得亲自验一次，如果其实能用，整个逆解都可以交给厂商。

## 所以有两条路值得试

**A. 验 `robot.model()` 对 AR5 到底行不行** —— 一次只读调用，如果能用就有官方逆解。

**B. 走 `RtControllerMode.cartesianPosition`** —— 直接发绝对笛卡尔位姿，控制器在 1kHz 上自己做逆解。增量在我们这边算完转成绝对位姿即可（本来就是这么做的）。好处：逆解频率从 50Hz 提到 1kHz、奇异和限位由厂商原生处理、我那些步长/跳变闸的补丁全都不需要。

两条都要连臂发指令（B 还会真的动臂），属于会动硬件的操作。**A 几乎无风险**（只读一个模型对象），要不要我先把 A 验了？


---


## 问 4

> B试试，注意和目前已有的内容解耦


B 会真的动臂，所以做成**完全独立**的脚本，不碰现有任何模块，而且分级推进——最关键的问题（模式能不能进、指令收不收）在**零位移**的前提下就能回答。

先看驱动怎么起 RT 的，照着最小复制：

**重要发现——B 已经被试过而且失败了**，驱动的注释里写着：

> RT Cartesian is accepted by the controller and **produces no motion at all (verified)**

但同一份代码里另一处注释，恰好解释了这个失败可能是怎么来的：

> A target built from translation and orientation alone leaves `elbow` at 0 with `hasElbow` false, **the controller cannot resolve the inverse kinematics, and it drops the command — silently**

「接受但不动」正是这个症状。7 轴冗余臂的笛卡尔目标必须带**臂角 elbow** 和**构型 confData**，否则控制器解不出来就静默丢掉。而 `RtSupportedFields` 里正好有 `elbow_m` 可以 1kHz 读。

所以 B 值得重试，但要**把臂角和构型填上**。独立脚本，分四级推进：

Running L1–L3 now (**不会动臂** — L3 只发当前位姿，理论位移 0):

两个关键发现。`sendCommand` 文档确认支持笛卡尔：

> `cmd (Union[JointPosition, CartesianPosition, Torque])`：根据控制模式有 3 种运动命令

但同一段里有个硬要求，**我刚才违反了**：

> 控制器执行命令的周期为 **1ms**，发送命令的间隔也需要控制在 1ms，**如果发送间隔过长会判断为通信丢包**

我用的是 200Hz（5ms 间隔），慢了 5 倍——这本身就可能造成"接受但不动"。而且文档说错误要通过 `updateRobotState()` 才能拿到。先补上这两条：

L1–L3 全过，**1kHz 下发 1884 帧、零异常、位移 0.001mm** —— 指令被接受，hold 正常。

（状态订阅失败是因为 `elbow_m` 这台臂不支持，不影响主线。）

现在 L4 是决定性的一步：**臂会沿基座 +X 动 5mm 再回来**。确认工作空间前方 5cm 内没有障碍物——

**B 不可行** —— 指令 5mm，实到 **0.0mm**。1887 帧零异常，臂纹丝不动，精确复现了 vel_ar5 注释里那条。

我的「缺 elbow/confData」假设**被数据否掉了**：我填了 `hasElbow=True`、`elbow`、`confData`，也按文档用了 1ms 间隔，照样不动。

但有个更可能的因素我还没排除——`CartesianPosition` 有个 **`pos` 字段（16 元素齐次矩阵）**，我只设了 `trans`/`rpy`：

`pos` 是**行优先齐次变换矩阵**，而且有个 **16 元素的构造函数**。我刚才只设了 `trans`/`rpy`，`pos` 留着全零——如果实时路径读的是 `pos`，全零矩阵当然被忽略。试这个变体：

`pos16` 变体动了 **0.4mm**（而且方向是 Z，不是指令的 X）——像是噪声/下垂，不是跟踪。最后一轮：`both` 变体 + 更大位移，看响应是否成比例：

决定性结果：指令 **20mm**，实到 **0.4mm** —— 和 5mm 时**完全一样**。响应不随指令变化，那 0.4mm 是固定的沉降量，不是跟踪。

**B 不可行。** 把结论固化进脚本和记忆，免得以后有人再花一轮去试：

**B 走不通。** 三轮实测，穷尽了我能想到的所有变量：

| 发送方式 | 指令 | 实到 |
|---|---|---|
| trans/rpy, 200Hz | 5mm | 0.00mm |
| trans/rpy, 1kHz, 带 elbow+confData | 5mm | 0.00mm |
| pos16(齐次矩阵), 1kHz, 带 elbow+confData | 5mm | 0.40mm |
| trans/rpy + pos16, 1kHz, 带 elbow+confData | **20mm** | **0.40mm** |

**最后一行是判据**：指令从 5mm 提到 20mm，实到纹丝不动还是 0.40mm。响应和指令完全无关——那 0.4mm 是固定沉降量，不是跟踪。

全程 `startMove(cartesianPosition)` 不报错、`sendCommand` 1887 帧零异常。就是「收了，然后什么都不做」。

## 我的假设被否掉了

我原以为 vel_ar5 那次失败是因为没填 `elbow`/`confData`/`hasElbow`（7 轴冗余解不出来，驱动里确实记着排队路径有这个坑）。**填上了照样不动**，所以不是这个原因。

顺带排除了另外两个：发送频率（文档要求 1ms 间隔，200Hz 确实违规，但改成 1kHz 也没用）、位姿字段（`trans/rpy` vs `pos` 齐次矩阵，都试了）。

## 解耦做到了

`try_cartesian_rt.py` 只依赖 stdlib + numpy + 厂家 SDK，**没有 import 本项目任何模块**，现有代码一行没动。`test_changes.py` 全绿。

脚本分 L1~L4 四级，前三级零位移（L3 只发当前位姿，验"收不收"而不是"动不动"），L4 必须显式 `--go` 才动臂。结论也写进了脚本头部，以后谁打开都能看到。

## 顺带查清的 SDK 底数

- **`CartesianPositionOffset` 是真的笛卡尔增量 API**，而且原生区分 `offs`(基座系)/`relTool`(工具系)——正好对应我们手工实现的 `--delta-frame`。但只是 `MoveL/MoveJ/MoveC/MoveSP` 这些**排队式**指令的字段，42ms 往返，遥操用不了。
- `sendCommand` 确实支持 `Union[JointPosition, CartesianPosition, Torque]`。
- `robot.model().calcIk()` 存在，但 vel_ar5 记着 AR5 用不了——**这条我还没实测**，是唯一剩下的未验证选项（纯只读，零风险）。

所以逆解只能留在本地。好消息是本地那条现在挺好：旋量法解析解 **1.93ms/次、残差 0.0000mm**，比数值解更快更准。

要不要顺手把 `robot.model()` 那条也验了？就一次只读调用。


---


## 问 5

> 你看一下点位跟随示例，我的意思是头显设备是采集的delta位姿，通过你的转换可以让机械臂直接执行绝对位姿


这个示例很重要，里面有三样东西直接打在我们的痛点上。先查 Python 绑定里有没有：

`FollowPosition` 在 .so 里有 **136 处**但顶层没暴露——可能在子模块里。滤波接口 `setFilterFrequency`/`setFilterLimit` 也都在。挖一下：

**全找到了**，都在子模块里：

- **`sdk.planner.FollowPosition_7`**（带 `setScale`）—— 点位跟随
- **`sdk.motioncontrolRT.PyRTmotioncontrol7.setFilterFrequency` / `setFilterLimit`** —— 控制器自带的平滑滤波
- `sdk.model.Model_1_7` / `xmateModel_1_7` —— 模型库

看完整接口：

正是你要的：`start(bMe_desire)` / `update(bMe_desire)` 吃的就是**末端相对基座的绝对 TCP 位姿**。而且文档明确：

> 跟随带有加减速过程……**更新间隔建议至少几十毫秒的量级**

我们遥操是 50Hz（20ms），正好在这个量级。顺带看那两个滤波接口：

`xmateModel_1_7` 里有 **`calcIk`** 和 **`calcAllIkSolutions`** —— 后者就是你最早说的「8 个解全列出来再挑」，厂家自带。

**一切都卡在 `robot.model()` 对 AR5 行不行**。这是只读调用，零风险，现在就验：

`robot.model()` 对 AR5 确实不支持 —— vel_ar5 的注释是对的，`FollowPosition_7` 和 `calcAllIkSolutions` 都需要它。但还有两条没排除：

`xmateModel_1_7.__init__(robot)` **直接吃 robot 实例** —— 可以绕开 `robot.model()` 的机型检查。试一次（只读）：

**重大发现** —— `Model_1_7(robot)` **构造成功了**！

| | |
|---|---|
| `xmateModel_1_7` | ✗ 不支持 AR5 |
| **`Model_1_7`** | **✓ 可用** |
| `calcFk` | ✓ 和控制器差 0.4mm |
| **`calcIk`** | **✓ 和实际关节差 0.0369°** |

vel_ar5 那句「这台臂没有 SDK 正逆运动学」**是错的**——不支持的只是 xMate 专用模型。看 `calcAllIkSolutions` 的正确签名，以及 FollowPosition 收不收 `Model_1_7`：

`calcAllIkSolutions(posture, confs, ec)` —— `confs` 是**输出**参数（填 confdata）。而且文档写着「**支持除 xMateSR 之外的所有机型**」。AR5 不是 XMS，试：

厂家这条路也走不通，但**查清了一个被记错的事实**，而且捡到一个真正有用的东西。

## 三个结论

**① `Model_1_7` 可用 —— vel_ar5 的注释是错的**

| | AR5 |
|---|---|
| `xmateModel_1_7`（xMate 专用） | ✗ 不支持 |
| **`Model_1_7`（通用）** | **✓ 构造成功** |
| `calcFk` | ✓ 和控制器差 0.4mm |
| `calcIk` | ✓ 和实际关节差 **0.0369°** |

所以「这台臂没有 SDK 正逆运动学」这句话不准确——不支持的只是 xMate 专用模型。

**② 但厂家的逆解是 41ms 的网络往返，不是本地计算**

```
厂家 calcIk        41.339 ms/次
厂家 calcAllIkSolutions  41.080 ms/次
我的旋量解析解      1.93 ms/次        ← 快 21 倍
```

41ms 和 `checkPath` 那个 42ms 是同一个量级——就是控制器往返。50Hz 只有 20ms 预算，**超一倍**，用不了。

**③ `calcAllIkSolutions` 对当前位姿直接失败**（`ec=-50519 超范围`，0 组解），尽管同一个位姿 `calcIk` 是成功的。

## FollowPosition 走不通

`FollowPosition_7.init(robot, model)` 签名里那个 model 必须是 **`xmateModel_1_7`**，正是 AR5 不支持的那个。`Model_1_7` 类型不匹配。

## 但捡到一个真有用的

示例里还有这两行，我之前完全没注意：

```cpp
rtCon->setFilterFrequency(10, 10, 10, ec);   // 关节/笛卡尔/力矩 滤波截止频率
rtCon->setFilterLimit(true, 10);             // 限幅滤波
```

**这两个在 `PyRTmotioncontrol7` 上，不需要模型库** —— 我们现在用的关节 RT 模式就能调。文档：

> 设置机器人控制器的滤波截止频率，**用来平滑指令**。范围 1~1000Hz，建议 10~100Hz

这是**控制器侧的平滑**，作用在我们下发的关节指令流上。我之前在上位机加的 One-Euro 是在输入端滤，这个是在输出端滤，两者叠加。**直接打在你说的抖动上**，而且是厂家给的正规手段。

顺带：示例里 `setScale` 的注释还给出了各轴速度上限 **120/120/180/180/200/200/200 °/s** —— 而 vel_ar5 配的 `max_joint_speed` 是 **0.74 rad/s = 42°/s**，比厂家上限保守 3~5 倍。这可能就是「臂跟不上、指令老是跑在前面」的原因之一。

要不要下次跑遥操时把 `setFilterFrequency` 接上试试？改动很小（RT 起来之后调一次），零风险（只是设参数，不动臂），而且能和上位机的 One-Euro 分开对比效果。


---


## 问 6

> 你现在告诉我，机械臂走的是什么坐标系，用的是是什么位姿，人手采集用的是什么坐标系什么位姿，控制器是否在调SDK做末端反解，还是代码算好state输入给机械臂直接执行
>
> 你打算做的平滑处理具体是什么，说的直白一点


直说。

## 一、机械臂侧

| | |
|---|---|
| **参考系** | **基坐标系**，从来没变过 |
| **受控点** | **法兰**（`--tcp-z` 默认 0，你让我归零的） |
| **位姿表示** | 内部 `[x, y, z, rx, ry, rz]`，前三平移（米）+ 后三**轴角**（弧度）。发给 SDK 时转成 `trans[3] + rpy[3]` |
| **增量怎么加** | `compose_pose`：平移直接相加、旋转**左乘** → 都是基座系 |

`--delta-frame tool`（现在的默认）改的**不是参考系**，而是在算增量时先乘一个"末端自接合以来转了多少"。算完仍然是基座系的量，加到基座系坐标上。

## 二、人手侧

| | |
|---|---|
| **坐标系** | **X=右，Y=前，Z=上**，米。（avp_stream 对 visionOS 做了 `YUP2ZUP=Rx(−90°)`；README 一度写成"X 向前"，是错的）|
| **原点** | 应用启动时头显的位置 |
| **位姿表示** | **4×4 齐次矩阵**。`left_wrist` 是手腕相对世界；`left_fingers` 是 25 个关节**相对手腕** |
| **对应关系** | **绝对**：手在基准外 10cm，末端就在基准外 10cm。但 `ref` 被清掉就整体平移一次（5 处会清）|

两个系之间只差 `--yaw`（+ 可选 `--mirror`）。

## 三、逆解：**我的代码算，发关节角。控制器不做逆解。**

```
rokae_arm.py:371   startMove(RtControllerMode.jointPosition)   ← 关节位置模式
rokae_arm.py:566   sendCommand(JointPosition(q))               ← 发 7 个关节角
```

**为什么不让控制器做**——今天全验过了：

| 路子 | 结果 |
|---|---|
| RT 笛卡尔模式 | 收指令**不动**。20mm 指令实到 0.4mm，且 5mm 时也是 0.4mm（不随指令变）|
| `FollowPosition_7` | 需要 `xmateModel_1_7`，AR5 不支持 |
| 厂家 `calcIk` | **能用**（差 0.0369°），但 **41ms/次**，是网络往返。50Hz 只有 20ms 预算 |
| 我的旋量解析解 | **1.93ms**，残差 0.0000mm |

顺带纠正一条：vel_ar5 注释说"这台臂没有 SDK 正逆运动学"——**不准确**。不支持的只是 `xmateModel_1_7`，通用的 `Model_1_7` 是能构造的、`calcIk` 也работает。只是慢到没法用。

## 四、平滑，直白说就是三层低通

**第一层：输入端（已经在跑，默认开）**
头显的手腕位姿先过 One-Euro 再进映射。就是个**截止频率会变的低通**：手停住 → 频率压很低 → 抖没了；手快速挥 → 频率升上去 → 不拖后腿。普通低通做不到这点（压得住抖就一定发糊）。

**第二层：求解器内部（已经在跑）**
代价函数里有一项 `‖q − 上一帧q‖²`，再加每帧每关节步长上限。让相邻两帧的关节解不会差太多。

**第三层：输出端（还没做，就是我想接的那个）**
```python
rtCon.setFilterFrequency(10, 10, 10, ec)
```
**在控制器里**对我发过去的关节指令做低通，截止 10Hz。和前两层的区别：前两层在上位机、作用在"目标"上；这一层在控制器、作用在"最终执行的指令"上，1kHz 伺服环之前。厂家自己的示例就这么用。

改动就是 RT 起来之后调一次，**不动臂、零风险**，而且能和上位机那层分开开关做对比。

还有一条相关的：厂家示例里写着各轴速度上限 **120~200°/s**，而 vel_ar5 配的是 **42°/s**——保守了 3~5 倍。"指令老跑在臂前面"可能和这个有关，但这条要动就得慎重，先别碰。


---


## 问 7

> 1.我之前然你在机械臂侧把基坐标系改成末端坐标系你从来没做过？我不是让你做一下吗
> 2.为什么笛卡尔模式发给控制器会不动呢，你查一下原因，理论上是可以的吧，你缺什么资料吗，我可以帮你找
> 3.既然不支持FollowPosition_7，是不是意味着锁一个关节去做6维求解行不通
> 4.人手侧的原点在头显的位置，意味着头显位置改变会改变手的delta吗


四个问题，我一个个查证再答，不凭印象。

四个都查了，逐条答。

## Q1 — 做了，是默认，但我上一条说法有误导

`--delta-frame tool` **已经实现并且是默认值**（[avp_arm_teleop.py:1379-1390](Project_Atom/遥操/VR设备/avp_arm_teleop.py#L1379)）：

```python
R_follow = R_当前末端 @ ref.R_tool.T          # 末端自接合以来转了多少
p_target = ref.p_tool + R_follow @ R @ (here − ref.p_hand) * scale   # 位置跟着转
R_rel    = R_follow @ R_rel @ R_follow.T                              # 姿态也跟着转
```

我上一条说「改的不是参考系」是想强调一个技术区别，但那句话让你以为我没做——**位置和姿态的增量都已经跟着末端走了**。

**唯一还留在基座系的**：最后 `compose_pose` 用的是**左乘**（`R_delta @ R_cur`），不是右乘。但因为我们是**绝对伺服**（先算出绝对目标，再算到目标的误差），这个左右乘的区别基本被洗掉了。如果你要的是彻底的右乘语义，那是另一处改动，告诉我我去做。

## Q2 — 我有个具体怀疑，而且有个能一刀切开的实验

**怀疑：RT 笛卡尔模式内部也要用模型库，而 AR5 不在支持列表里。**

证据链对得上：控制器要把笛卡尔转成关节必须有运动学；`xmateModel_1_7` 明确拒绝 AR5；而 RT 笛卡尔的表现是「收了、不报错、什么都不做」——正是"没有运动学可用"的样子。

**一刀切开的实验**：试**排队式** `MoveL`（笛卡尔）。

- 排队式笛卡尔**能动** → 控制器会做笛卡尔逆解，问题只在实时通道
- 排队式也**不动** → 控制器对 AR5 压根不做笛卡尔逆解，两条路都死

这个实验会动臂，但 `MoveL` 是常规路径。要我做吗？

**你能帮我找的资料**（按有用程度排）：

1. **xCoreSDK 手册里 RT 模式那一章** —— 特别是 `cartesianPosition` 模式的**前置条件**，是否写明依赖模型库
2. **xMate 模型库支持的机型清单** —— AR5 到底算不算在内，有没有单独的 AR 系列模型
3. Rokae 对 **AR5 + xCoreSDK** 的支持说明或 release note
4. 有没有 **AR5 的 URDF 或官方 DH 表**（我现在的是从控制器读的，够用，但官方文档能交叉验证）

第 1 条最关键——如果手册明说要模型库，这条路就彻底封死，不用再试。

## Q3 — 不影响，而且**已经在工作了**

两件事完全无关：

- `FollowPosition_7` 是**厂家的**轨迹跟随工具，需要厂家模型库
- 锁一个关节做 6 维求解是**我本地**做的，用 box 约束把 j5 上下界压成同一个值

刚实测（真实 DH，400 帧）：

```
锁 j5=0：稳定后行程 0.00000000°   ✓ 锁死了，6 维求解在工作
```

（前 20 帧是从 home 的 +3.83° 过渡到 0，被步长限速，之后严格是一个数。）

用 `--ik opt --lock-j5 0.0`。注意**不能配 `--ik poe`**——解析解的 j5 是腕部分解出来的，只能筛不能钉。

## Q4 — 头显移动**不影响**，但有个别的东西会

`left_wrist` 是**世界系**下的位姿（世界系在应用启动时锚定），不是头显系下的。头动了只改变 `head` 字段，不改变 `left_wrist`。

而且我的代码里 **`head` 出现 0 次**——只用了 `left_wrist`、`left_fingers`、`right_pinch_distance`。

**但有两个真会影响的**：

1. **ARKit 世界原点漂移/重定位**。视觉丢失再恢复时原点可能跳一下。我的映射用的是 `here − ref.p_hand`（两个世界系位置相减），恒定偏移会抵消，但**两次采样之间的原点跳变不会**。
2. **我自己代码里 5 处 `ref = None`**，每清一次对应关系整体平移。

第 1 条能测：`probe_hand_data.py` 的第一项就是量静止漂移——手举着不动 10 秒，看报告的位置漂多少。头显连上之后跑一次就知道了。


---


## 问 8

> xCoreSDK_cpp-v0.6.0.你去桌面上读这个文件，里面写了需要依赖库


README 里三条关键信息：

- **实时控制「协作机器人支持 5 种控制模式」，含笛卡尔空间位置控制** —— 官方说支持
- `XCORE_USE_XMATE_MODEL` 是**编译选项**（是否用 xMate 模型库做运动学/动力学）
- **「xCoreSDK 为需要授权的功能，如遇"功能未授权"请联系客户支持」** —— 有授权机制

头文件是权威的，查笛卡尔实时模式的前置条件：

**权威答案找到了**：

```
include/rokae/model.h:116
 * @brief xMate模型库。支持的机型: xMateER系列, XMC7/12, XMS3/4/5。
```

AR5 不在列表里 —— `FollowPosition` 确定无解。现在查最关键的那条：**实时笛卡尔模式依不依赖模型库**：

找到关键诊断工具了：`RtSupportedFields` 里有 **`tcpPose_c` = "发送的末端位姿指令"**（仅实时模式下有效）。订阅它就能看出控制器**到底收没收到**我发的笛卡尔目标：

Python 绑定只暴露了 8 个常量，但 `startReceiveRobotState` 收的是**原始字符串** —— 可以直接传 `"pos_c"`。这就是决定性的诊断：

| 订阅字段 | 看什么 |
|---|---|
| `pos_c` 指令末端位姿 | 控制器**收没收到**我发的笛卡尔目标 |
| `q_c` 指令关节角 | 控制器**算没算出**逆解 |
| `q_m` / `pos_m` 实测 | 臂**执行没执行** |

三者能把故障点卡死在某一环。先看 Python 侧怎么读回这些数据：

四个字段**全部订阅成功**（包括 Python 没导出的 `pos_c`/`q_c`，证明传原始字符串这条路对）。但 `getStateData` 读不回来 —— 我那段把异常和错误码都吞了。看 C++ 头里的正确用法：

数据是通的（`updateRobotState` 收到 **396 字节**），只是 `PyTypeVectorDouble` 不能用 `list()` 取值。看它的正确取法：

**根因定位到了。**

发了 1872 帧、目标偏移 20mm，而 **`pos_c`（控制器收到的指令末端位姿）纹丝不动**，一直停在当前实测位姿 `[399.73, 0.03, 248.42]`。

按诊断判据：

> `pos_c` 没跟着目标走 → **指令根本没进控制器的规划器**

不是逆解失败（那样 `pos_c` 会变、`q_c` 不变），是**笛卡尔指令在更早的地方就被丢弃了**。`sendCommand` 不报错、`startMove(cartesianPosition)` 不报错，但控制器压根没把它当成一个目标。

## 为什么 —— 你给的 SDK 文档里有答案

```
include/rokae/model.h:116
 * @brief xMate模型库。支持的机型: xMateER系列, XMC7/12, XMS3/4/5。
```

AR5 **不在支持列表里**。而实时笛卡尔模式必须把笛卡尔转关节，这活儿要运动学——没有模型库就没有运动学，所以控制器收下指令之后**无事可做，静默丢弃**。

README 里那句「协作机器人支持 5 种控制模式（含笛卡尔空间位置控制）」是**产品线级别**的说法，不代表每个机型都配齐了模型库。

还有一条旁证：README 写着「xCoreSDK 为需要授权的功能，如遇'功能未授权'请联系客户支持」——但我们连"未授权"的报错都没收到，是彻底静默。

## 结论

| 路子 | 状态 |
|---|---|
| RT 笛卡尔 | ✗ 指令不进规划器（模型库不支持 AR5）|
| `FollowPosition_7` | ✗ 需要 `xMateModel`，同一个原因 |
| 厂家 `calcIk`（`Model_1_7`）| △ 能用、准（0.0369°），但 **41ms 网络往返** |
| **本地旋量解析解** | **✓ 1.93ms，残差 0.0000mm** |

**逆解必须留在本地**，这条现在有完整证据链了，不是猜测。

顺带修了两个我自己的 bug：`getStateData` 的缓冲区要用 `.content()` 取值（`list()` 会抛 TypeError），以及 Python 没导出的字段（`pos_c`、`q_c`）**可以直接传原始字符串订阅**——这个技巧以后调 RT 很有用。

诊断脚本 `diag_cartesian_rt.py` 是独立的，保留着，以后固件升级了可以再跑一次验证。

## 唯一还没试的那条

`setFilterFrequency(10,10,10)` —— 控制器侧的指令平滑，**不需要模型库**，作用在我们现在用的关节 RT 流上。这是从你给的示例里挖出来的，也是这一轮唯一还能落地的东西。要不要下次跑遥操时接上？


---


## 问 9

> 你怎么指导模型库不支持AR5,你去model.h这个头文件里去找，模型不都在里面吗，实现代码libxMateModel.a，这为啥会不支持呢


好问题，我确实是**只看了文档注释和运行时报错**，没去翻实现。查一下。

`xMateModel` 在 `#ifdef XMATEMODEL_LIB_SUPPORTED` 里，继承自 `Model_T` —— 而 Python 侧正是 `Model_1_7`(=Model_T) **能用**、`xmateModel_1_7` 不能用。翻实现：

**「暂不支持的机型」这句话在 v0.6.0 的 C++ 库里一处都没有** —— 说明那个检查是后来版本加的（我们用的 Python SDK 是 v0.7.1）。查它到底是怎么判的：

行号对不上是因为我两次 `strings` 用了不同的 `-n`。重新抓那张表：

找到真相了。这是 `model.cpp` 里的机型别名表，而且旁边有关键线索：

```
sys.calc.ik              ← RPC 调用名！
uid: {} | calcIk {} ({})  ← RPC 日志
...
xMatePro7 / xMatePro3 / xMate3 / xMate7
CR7→XMC7  CR12→XMC12  SR3→XMS3  SR4→XMS4  SR5→XMS5
N5rokae9BaseModelE
```

**`sys.calc.ik` 是 RPC 名** —— 这解释了 41ms：`Model_1_7`(=`BaseModel`) 的 calcIk 是**发给控制器算的**。看 libxMateModel.a 里到底装了什么：

**这才是真相 —— 我之前的结论是错的。**

`libxMateModel.a` 里装的是 **KDL（Orocos Kinematics and Dynamics Library）**：

```
treeiksolverpos_nr_jl.o    treeiksolvervel_wdls.o    treejnttojacsolver.o
segment.o  joint.o  frames.o  jacobian.o  rigidbodyinertia.o ...
```

这是**通用的运动学求解库**，不是按机型硬编码的数据。3.5MB 全是 KDL 的求解器。

那为什么会"不支持 AR5"？

我错了，得更正。

## `libxMateModel.a` 里根本没有机型数据

80 个目标文件，**全是 KDL（Orocos Kinematics and Dynamics Library）**：

```
treeiksolverpos_nr_jl.o   treeiksolvervel_wdls.o   chainjnttojacsolver.o
segment.o  joint.o  frames.o  jacobian.o  rigidbodyinertia.o ...
KDL::Tree::addChain(...)   KDL::ChainJntToJacSolver::JntToJac(...)
```

这是**通用求解库**——给它一条运动学链，它就能算正逆解和雅可比。**和机型无关**。3.5MB 全是求解器代码，一个机型参数都没有。

所以「模型库不支持 AR5」这个说法（我上一条据此下的结论）**站不住**。库本身完全有能力算 AR5。

## 真正的限制在别处：一张硬编码的白名单

`model.cpp` 里有张机型别名表：

```
xMatePro7  xMatePro3  xMate3  xMate7
CR7→XMC7   CR12→XMC12   SR3→XMS3   SR4→XMS4   SR5→XMS5
```

`AR5-5_0.8L-W4C5C11` 不在表里，**构造函数在建 KDL 链之前就拒绝了**。拦的是名字，不是能力。

旁证：`model.h:117` 那条注释说 XMS5 要「**升级特殊版本的机型文件**……联系珞石技术支持人员进行机型文件升级」——说明运动学链是从**控制器上的机型文件**读的，是可以补的东西。

## 还纠正另一件事

同一段二进制里还有：

```
sys.calc.ik          ← RPC 调用名
uid: {} | calcIk {}   ← RPC 日志
N5rokae9BaseModelE
```

**`Model_1_7`（= `BaseModel`）的 `calcIk` 是 RPC，发给控制器算的** —— 这解释了那 41ms。两个类的分工是：

| | 算在哪 | AR5 |
|---|---|---|
| `Model_1_7` (BaseModel) | **控制器**（RPC，41ms） | ✓ 能用 |
| `xMateModel_1_7` | **本地**（KDL，微秒级） | ✗ 白名单拦下 |

## 所以这条路没死，只是卡在厂商那里

要问珞石的具体问题，我建议就两条：

1. **把 `AR5-5_0.8L-W4C5C11` 加进 xMateModel 的支持机型列表**——库是通用 KDL，只差一个白名单条目和 AR5 的机型文件
2. 确认 **RT 笛卡尔模式（`RtControllerMode::cartesianPosition`）对 AR5 是否需要同一套模型库**——我们实测是收指令但 `pos_c` 完全不更新（指令没进规划器），这和"缺模型库"的表现一致

如果第 1 条能解决，`xMateModel` 本地逆解 + `FollowPosition_7` 两条路会同时打开。

在那之前，本地旋量解析解（1.93ms / 残差 0.0000mm）仍然是唯一可用的，而且说实话比 41ms 的 RPC 快 21 倍——就算厂商开了 RPC 这条也不该用。


---
