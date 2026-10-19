# 仿真参考文献：单机视觉语义运动

这些 PDF 用来约束后面的 Isaac Lab / LocoLab 仿真，不改本周范围。`Papers.rar` 与单独给出的 RLOC PDF 是同一篇。作者名只写从 PDF 正文里能读出的；综述首页的作者栏没有被文本抽出来，所以那篇只标出处。

本周仿真仍然是：一台 Go2，仿山洞走廊，`dry_rock` / `wet_slippery`，接触前把 `[p_dry, p_wet, confidence, transition_distance]` 送进 MLP actor。高度扫描保留。PPO 文件不改。

## 仿真时必须守住的四条

1. **接触前和接触后是两件事。** Miki 等人（Science Robotics, 2022）写明：只靠本体感觉，机器人必须先踩到地面才能改步态，速度因此上不去。IEEE RAM 2026 的 perceptive locomotion 综述把同一句话写成：blind policy 对地形变化只能做 post-contact reaction。我们的 Oracle / Vision 必须在足端碰到新材料之前改变动作。HIM、PrePARE、ProNav、Sanchez-Delgado 测的是踩到之后或打滑过程中的本体反应，用来做对照，不拿来当 actor 的视觉输入。
2. **视觉类别必须对应真实摩擦差。** Wensing 等人（T-RO, 2024）把可行接触写成摩擦锥和单边接触。干岩和湿滑岩几何可以接近，但 PhysX 里的摩擦必须不同，并用同一速度命令比较滑移。纹理不一样而摩擦一样，不能叫 semantic locomotion。
3. **训练顺序用特权教师，再换成真实传感器。** Ha 等人（IJRR, 2025）把 privileged learning 写成标准做法：教师先看仿真里的真值，学生再改成可部署的传感器。这就是计划里的 Oracle，然后冻结视觉网络做 Vision fine-tune。
4. **置信度要能关掉错误语义。** PIETRA 用不确定性决定何时相信学习模型、何时退回物理模型。本周不训练 evidential network。对应实现是：置信度裁剪到 `[0, 1]`，fine-tune 时丢掉或打乱语义向量，评估时保留 Shuffled。

## 各篇怎么用

| 文献 | 它实际解决的问题 | 进入本周仿真的部分 | 留到以后 |
|---|---|---|---|
| Miki et al., Sci. Robot. 2022. Learning robust perceptive locomotion for quadrupedal robots in the wild | 野外四足要在接触前融合本体和外部地形，否则只能摸着走 | 成功标准：同一条洞内路线上，Vision 在第一次接触前的动作和 `Slip_first` 优于 Baseline。场景里保留他们点名的失效条件：低光照、反光积水、几何看起来像平地但表面滑 | 不复现 attention recurrent elevation encoder，也不做端到端像素 PPO |
| Ha et al., IJRR 2025. Learning-based legged locomotion: State of the art and future perspectives | 学习式足式运动的 MDP、仿真和教师—学生训练该怎么配 | 观测分成本体、速度命令、height scan、语义向量；奖励里保留速度跟踪，另加接触滑移；动作仍是 12 个关节目标 | 不扩到人形，不改成模仿学习或 DAgger |
| Long et al., ICLR 2024. Hybrid Internal Model | 只用关节编码器和 IMU，把摩擦和高度图当成扰动，从机器人下一步响应里估出来 | 作为反例写进实验说明：HIM 的摩擦信息来自接触后的响应。我们的 Baseline 就是这条线的弱版本 | 不把 HIM encoder 接进 LocoLab，也不改他们那一版 rsl-rl |
| Wensing et al., T-RO 2024. Optimization-based control for dynamic legged robots | 在线最优控制如何处理接触、摩擦锥和模型简化 | 材料合同：刚体、单边接触、库仑摩擦。湿滑只降低 μ，不做成可变形地面。Mac 上的 `slip_proxy` 只排序，GPU 上要用接触切向速度验收 | 本周控制器仍是 PPO，不上 whole-body MPC |
| Gangapurwala et al. RLOC: Terrain-aware legged locomotion using RL and optimal control | RL 把本体、外部感知和速度命令映射成落足，再用模型控制器跟踪 | 感知特征和跟踪政策分开。语义向量是计划量，actor 仍是速度跟踪器 | 不加落足规划器和 ANYmal 的 whole-body tracker |
| Corbères et al., IEEE Access 2025. Perceptive locomotion through whole-body MPC and optimal region selection | 深度相机高程图上选摩擦锥能包住重力的落足区域，再用 Crocoddyl 跟踪 | 落足区域必须在接触前按摩擦筛选。两类地面是这个区域选择的离散版：湿滑区不是“看起来不同”，而是摩擦锥更窄 | 不跑 Crocoddyl，不做连续落足优化 |
| Dey et al. PrePARE: Predictive proprioception for agile failure event detection | 石灰岩洞穴等湿滑地面上，只用本体预测打滑，提前最多约 720 ms 切到 crawl / amble | 评价指标可以对照：他们预测的是跌倒前的打滑，我们记录的是新材料接触后 0.2 s 的足端水平速度。场景气质可参考洞穴湿滑，几何仍用我们的等高走廊 | 不根据本体切 gait mode，也不把 720 ms 写成我们的指标 |
| Elnoor et al. ProNav | 用关节位置、力和电流估计可通行性，再和外部感知一起选步态 | 日志分开存：RGB 预测即将接触的类；接触力和足端速度证明这个类在物理上成立 | 不估计泥地陷入、缠绕或能耗最优步态 |
| Cai et al. PIETRA | 视觉可通行性在分布外会错；用证据不确定性把预测交回物理模型 | `confidence` 低时语义向量不应主导动作。Shuffled 用来证明策略确实在用语义 | 不做 evidential NN、CVaR 或 MPPI |
| Sanchez-Delgado et al. Towards proprioceptive terrain mapping | 相机和雷达能给几何和语义，给不了滑移、能耗和稳定裕度；本体感测补这一层 | 语义来自 RGB，滑移来自接触。两类材料先不要做成多层 2.5D 地图 | 月面重力、永久阴影区、在线地形图都不在本周 |
| Kang et al. RL + model-based control using on-demand optimal control | 有限时域最优控制产生参考动作，RL 模仿；粗糙地面使用机身周围 7×11 height scan | height scan 保持 LocoLab 原来的局部扫描，语义是额外分支。动作变化率惩罚继续用现有奖励 | 不在线生成最优控制参考轨迹 |
| Zargarbashi et al. MetaLoco | 用 meta-RL 和动作模仿让一套政策零样本换机器人；域随机化包含地形摩擦和执行器延迟 | 每一类内部可以随机化摩擦和外观：干岩围绕高 μ，湿滑岩围绕低 μ，再加光照和纹理。类别之间的 μ 差距必须留住 | 不做跨机器人 meta-RL，不换机体 |
| IEEE RAM, March 2026. Learning perceptive legged robot locomotion in the real world: A systematic review | 只评学习式、接触前的感知运动。应用分成不规则地面、不连续地面、避障、与障碍交互 | 我们落在第一类里的摩擦变化，而且几何连续。综述引用的 Brandão et al., Humanoids 2016，已经用材料识别 CNN 给双足在湿滑地面上做层级规划；本周是这个想法在四足速度跟踪里的仿真版 | 不做台阶、缝隙、避障和全身过障碍 |

## 对现有实验设计的直接约束

- **Baseline** 对应综述里的 post-contact / blind：有 height scan 和本体，没有语义。
- **Oracle** 对应 Ha 综述的 teacher：actor 收到仿真真值 `[p_dry, p_wet, confidence=1, distance]`。
- **Vision** 对应 Miki 的接触前外部感知，但外部量是 RGB 语义，不是高程图。高程仍走原来的 height scan。
- **Shuffled** 对应 PIETRA 的“错误感知不应保留收益”。错类别若仍然少滑，就不能声称策略使用了语义。
- 山洞外观参考 Miki 的地下洞穴和 PrePARE 的石灰岩湿滑，物理只实现两种刚体摩擦。碎石、泥、软地面留在论文的未来工作里，不写进第一版场景。

## 读的时候要留的异味

- Miki 的野外成功率建立在高程图加大量真实试验上。我们一周的仿真不能写成“野外感知运动已经解决”。
- HIM 的“任意地形、任意扰动”来自接触后的响应估计。把它的敏捷性曲线拿来当我们的视觉收益，比较对象就错了。
- RLOC 和 Corbères 的地形感知后面还有模型跟踪器。只复现他们的 RL 前半段，不能声称达到了他们的落足稳定性。
- PIETRA 和 ProNav 的可通行性是导航代价，不是 50 Hz 关节政策。置信度可以学他们的用法，规划器不搬。
- Wensing 综述讨论的是最优控制里的接触模型。PhysX 摩擦参数通过验收，并不等于摩擦锥约束被写进了控制器。
