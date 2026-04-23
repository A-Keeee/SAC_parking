# 多车协同泊车环境改造 Walkthrough

我们已经成功将 HOPE 单车泊车仿真拓展为 **三车协同泊车** 环境，并实现了关键指标的统计与可视化。

## 1. 核心配置更新 (`configs.py`)
为了支持多车环境，我们在 `src/configs.py` 中新增了以下多智能体参数：
- `NUM_AGENTS = 3`: 设定环境默认包含的协同车辆数。
- `MULTI_CAR_COLORS`: 为 3 辆车分配了不同的醒目颜色（道奇蓝、珊瑚色、石灰绿），便于可视化追踪。
- `SAFETY_DIST_THRESHOLD = 1.0`: 设定了车辆间的最小安全距离阈值 (1.0m)。
- 调整了窗口尺寸 `WIN_W_MULTI` 和 `WIN_H_MULTI`，使其有足够空间展示三辆车。

## 2. 仿真地图升级 (`parking_map_normal.py`)
我们在原有的单车位生成逻辑外，新增了 `generate_multi_car_bay_parking_case` 和 `generate_multi_car_parallel_parking_case`：
- **目标车位生成**: 场景会生成更长的停车区域（12个车位）。引擎随机抽取 3 个互不重叠的车位作为 3 辆车各自的目标车位。未被选中的车位有 50% 概率生成静态障碍车辆。
- **随机起始位置**: 在后方广阔的起步区域中，引擎会为每辆车生成初始状态。我们引入了多重碰撞检测，确保这 3 个起始位置 **互不重叠**，并且不与任何目标车位或静态障碍物冲突。
- **状态列表化**: `ParkingMapNormal` 的核心输出由单一 `State` 对象变更为 `List[State]`，以兼容并初始化多辆车。

## 3. 多智能体环境包装 (`car_parking_base.py` & `env_wrapper.py`)
我们通过继承和复写，构建了 `MultiCarParking` (以及外壳 `MultiCarParkingWrapper`)：
- **车辆与传感器实例化**: 环境初始化时同时创建 3 个 `Vehicle` 和 3 个 `LidarSimlator` 对象，每辆车拥有独立轨迹跟踪、车辆模型（Kinematic Single-Track Model）和激光雷达探测系统。
- **协同奖励函数 (`_get_reward`)**: 奖励函数从单车扩展到了多车，并增加了两种特色奖励机制：
  1. **协同时间奖励 (`cooperative_time_reward`)**: 以最慢那辆车的时间作为系统性能基准，所有车辆成功停放后，会根据总体用时发放团队奖励。
  2. **智能体安全性奖励 (`safety_reward`)**: 利用新加入的 `_get_safety_reward`，实时计算三车两两之间的最短距离。如距离小于 `SAFETY_DIST_THRESHOLD`，系统给予严重处罚；若发生车间碰撞，则执行极高惩罚。
- **步进演进 (`step`)**: 接受含有 3 个控制动作的列表 `actions`。针对每辆车进行物理仿真演算，分别计算其碰撞、出界和到达状态。如果某一车辆已顺利泊入目标位，它将停止动作；环境持续演进直至所有车均结束。

## 4. 评估与可视化大屏 (`visualize_multi_car.py`)
为了满足可视化 3 辆车同时泊车以及呈现关键指标的要求，我们在 `src/evaluation/` 中新增了专用脚本 `visualize_multi_car.py`。
- **渲染界面**: 依赖 Pygame，可以动态展现这 3 辆车如何从不同的起点向不同的目标车位移动。每个目标车位的边框颜色与对应车辆颜色相匹配。
- **关键性能指标 (KPI) 统计**: 
  - **成功率**: 统计每回合是否做到 3 车全部抵达，或部分抵达/失败。
  - **平均时间**: 计算成功回合下，3 车泊车完成的最大协同用时。
  - **安全性**: 记录每回合仿真中车辆间贴得最近的瞬时距离，反映决策模型的安全裕度。
- **图表自动生成**: 评估结束后，代码会利用 `matplotlib` 生成上述 KPI 的分布直方图，保存在 `log/eval_multi/` 目录下，让你直观把握模型的综合协作水平。

> [!TIP]
> 运行测试和可视化的命令：
> `python src/evaluation/visualize_multi_car.py --eval_episode 10 --visualize True`
> 如果有训练好的 PPO/SAC 多智能体模型路径，可通过 `--ckpt_path <your_model.pt>` 参数加载。

---
**附：改动文件一览**
- [configs.py](file:///home/ake/1-500/HOPE_ge/src/configs.py)
- [parking_map_normal.py](file:///home/ake/1-500/HOPE_ge/src/env/parking_map_normal.py)
- [car_parking_base.py](file:///home/ake/1-500/HOPE_ge/src/env/car_parking_base.py)
- [env_wrapper.py](file:///home/ake/1-500/HOPE_ge/src/env/env_wrapper.py)
- [visualize_multi_car.py](file:///home/ake/1-500/HOPE_ge/src/evaluation/visualize_multi_car.py) (New)
