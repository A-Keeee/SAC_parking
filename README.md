# HOPE: 多车协同泊车仿真环境 (Multi-Car Cooperative Parking)

本项目在原始的 [HOPE](https://arxiv.org/abs/2405.20579) 单车泊车规划器基础上进行了**多智能体（Multi-Agent）环境扩展**，实现并支持了**三辆车同时在复杂场景下进行协同泊车**。

此扩展保留了原版强化学习结合 Reeds-Shepp 曲线的混合策略架构，并在环境底层、奖励函数以及数据可视化层面进行了全面升级。

---

## 🌟 核心特性升级

1. **并行多智能体环境 (`MultiCarParking`)**：
   - 支持多辆车（默认 `NUM_AGENTS = 3`）在同一个场景中同时控制与演算。
   - 包含多台车独立的物理引擎（基于运动学单轨模型）与 Lidar 模拟器。
2. **防碰撞与无重叠生成机制**：
   - 自动生成更宽阔的停车场地。
   - 保证 3 辆车的**初始起步位置绝对互不重叠**，并且避开障碍物区域。
   - 保证 3 辆车的**目标车位绝对互不相邻且不重叠**，符合现实场景的散落车位分布。
3. **强化学习协同奖励**：
   - **协同总时间奖励**：在全队均完成泊车时，基于耗时最长的那辆车的时间给予总奖励，鼓励团队整体尽早完成任务。
   - **智能体间安全性奖励 (Inter-Agent Safety)**：引入了新的车间安全距离阈值 (`SAFETY_DIST_THRESHOLD = 1.0`米)。当车辆间距低于阈值时，引擎会实时给予惩罚，严惩车间碰撞，鼓励车辆互相让行。
4. **全面的指标可视化评估**：
   - 采用 Pygame 渲染带有颜色区分（例如：蓝色、珊瑚色、绿色）的 3 辆车及它们各自对应的目标车位。
   - 脚本结束自动保存性能指标统计图表：
     - **停车成功率分布图**（记录 0辆、1辆、2辆、3辆同时停入的概率）
     - **协同用时直方图**（成功回合中的最大耗时分布）
     - **安全间距直方图**（测试每回合车辆靠得最近的极限距离）

---

## 🎯 奖励函数深度重构 (Reward Function Redesign)

针对多车协同训练初期容易出现的“原地打转”及难以收敛等问题，我们对底层的奖励函数进行了深度重构与平衡：

1. **引入 Reeds-Shepp 距离约束 (`rs_dist_reward`)**
   - **问题**：原有的纯欧氏距离 (`dist_reward`) 忽略了车辆的非完整运动学约束（Non-holonomic constraints），导致模型在复杂姿态下无法找到可行路径。
   - **优化**：启用了基于 Reeds-Shepp 曲线的距离计算（权重调整为 `5`），这使智能体能够更准确地评估当前位姿到目标车位的“真实驾驶距离”。
2. **强化航向角对齐 (`angle_reward`)**
   - **问题**：原有的角度奖励权重为 0，导致车辆在靠近车位前毫无对齐意识，容易在目标附近盲目乱转。
   - **优化**：增加了航向角度奖励（权重为 `2`），引导车辆在行驶过程中尽早调整车头/车尾朝向。
3. **新增动作惩罚 (`action_reward`)**
   - **问题**：过去对激进的转向没有限制，导致智能体容易通过“极限打满方向盘原地打转”来规避安全惩罚或时间惩罚。
   - **优化**：在 `env_wrapper.py` 的 `reward_shaping_multi` 中引入了基于方向盘转角二次方的动作惩罚（`action_reward = -(steer / max_steer)^2`），促使智能体学习更平滑、拟人的驾驶轨迹。
4. **平滑化安全距离惩罚 (`safety_reward`)**
   - **问题**：原版安全惩罚采用断崖式的硬阈值，容易导致智能体“受惊”并在多车靠近时直接停滞。
   - **优化**：将其修改为基于距离的连续指数衰减惩罚（`exp(-3 * min_dist / SAFETY_DIST_THRESHOLD)`）。这种“势场”式的平滑梯度能让智能体更容易学习到“互相避让”的安全策略，从而打破原地打转的僵局。

---

## 🛠️ 安装与配置

请确保你已经安装了基础版本所需的依赖。

```bash
git clone https://github.com/jiamiya/HOPE.git
cd HOPE
conda create -n HOPE python==3.8
conda activate HOPE
pip3 install -r requirements.txt
```
*(注意：请根据系统环境自行到 [PyTorch 官网](https://pytorch.org/) 安装合适的 `torch` 版本)*

所有的多车常量配置都在 `src/configs.py` 中，你可以根据需要自由修改：
```python
NUM_AGENTS = 3                    # 协同车辆数量
SAFETY_DIST_THRESHOLD = 1.0       # 车间最小安全距离(m)
COOPERATIVE_TIME_WEIGHT = 2.0     # 协同时间奖励权重
SAFETY_REWARD_WEIGHT = 3.0        # 安全距离惩罚权重
```

---

## 🚀 运行与评估测试

我们在 `src/evaluation/` 目录下新增了专属的多车评估与可视化脚本。

### 1. 运行多车随机策略测试 (可视化观看)
如果你没有加载任何模型，代码将展示一个初始的随机运动状态：
```bash
cd src
python ./evaluation/visualize_multi_car.py --eval_episode 10 --visualize True
```
> **提示**: `visualize True` 开启实时画面渲染，你将看到 3 辆车从随机起点寻找自己的目标位置。运行结束后，脚本会在 `log/eval_multi/<timestamp>/` 下生成相关的 KPI 性能评估图表。

### 2. 运行预训练模型测试
该多车仿真环境依然兼容单车环境的 action 空间（通过策略共享）。你可以直接传入之前的 `SAC` 或 `PPO` 单车模型，三辆车会复用相同的预训练大脑进行控制：
```bash
cd src
python ./evaluation/visualize_multi_car.py --ckpt_path ./model/ckpt/HOPE_SAC1.pt --eval_episode 100 --visualize True
```

---

## 🧠 训练说明

目前，我们基于单车参数共享（Parameter Sharing）机制编写了专用于多车环境的训练脚本。3 辆车在仿真环境中产生的交互数据会被收集到同一个经验回放池（Replay Buffer）中，并共享同一个神经网络进行更新，这种方式能显著加快收敛，并且天然兼容原有的单车权重。

你可以通过以下命令启动多车训练（从零开始或微调）：
```bash
cd src
# 从零开始训练多车 SAC：
python ./train/train_HOPE_sac_multi.py

# 在多车环境中微调单车预训练模型（利用单车的先验知识）：
python ./train/train_HOPE_sac_multi.py --agent_ckpt ./model/ckpt/HOPE_SAC0.pt --visualize True
```
```
> **注意**：训练中会在 `log/exp/sac_multi_<timestamp>/` 下保存 Tensorboard 日志和最优模型。评估时请将 `--ckpt_path` 指向这里生成的 `.pt` 权重。

---

## 📄 评估图表说明

当你执行完 `visualize_multi_car.py` 脚本后，前往 `log/eval_multi/` 目录，你将看到以下 3 张重要的数据分析图：
- `success_distribution.png`: 柱状图展示在 N 个回合中，所有车全进、进 2 辆、进 1 辆的比例。
- `cooperative_time.png`: 多车协作时间的分布状况。
- `safety_distance.png`: 车辆间极限距离的统计，评估当前 AI 的驾驶风格是否“激进”或“容易发生事故”。
