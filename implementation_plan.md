# 多车协同泊车环境改造方案

将现有 HOPE 单车泊车仿真拓展为 **3 辆车同时泊车** 的多智能体协同环境。

## 改造范围总览

| 模块 | 文件 | 改动类型 |
|------|------|----------|
| 配置 | `configs.py` | MODIFY |
| 地图生成 | `parking_map_normal.py` | MODIFY |
| 仿真环境 | `car_parking_base.py` | MODIFY |
| 奖励包装 | `env_wrapper.py` | MODIFY |
| 可视化 | `evaluation/visualize_multi_car.py` | NEW |

---

## Proposed Changes

### 1. 配置模块

#### [MODIFY] [configs.py](file:///home/ake/1-500/HOPE_ge/src/configs.py)

新增多车相关配置常量：

```python
# Multi-car settings
NUM_AGENTS = 3                    # 同时泊车的车辆数
MULTI_CAR_COLORS = [
    (30, 144, 255, 255),   # dodger blue  - Car 0
    (255, 127, 80, 255),   # coral        - Car 1
    (50, 205, 50, 255),    # lime green   - Car 2
]
SAFETY_DIST_THRESHOLD = 1.0       # 车间安全距离阈值(m)
COOPERATIVE_TIME_WEIGHT = 2.0     # 协同时间奖励权重
SAFETY_REWARD_WEIGHT = 3.0        # 安全性奖励权重
WIN_W_MULTI = 800                 # 多车场景窗口宽度
WIN_H_MULTI = 600                 # 多车场景窗口高度
```

---

### 2. 地图生成器（多车位+多起点）

#### [MODIFY] [parking_map_normal.py](file:///home/ake/1-500/HOPE_ge/src/env/parking_map_normal.py)

**核心改动**：`ParkingMapNormal` 类从管理单个 `start/dest` 扩展为管理 `starts[]/dests[]` 列表。

**设计方案**：
- 扩大停车场尺寸（`bay_half_len` 从 15 增大到 25），为 3 辆车提供足够空间
- 在一排车位中随机选取 **3 个不相邻的** 目标车位（保证车位间至少隔一个空位，避免目标车位过于拥挤）
- 为每辆车生成随机起始位置，确保：
  - 起始位置之间不重叠（`start_box_i.intersects(start_box_j) == False`）
  - 起始位置不与任何障碍物/目标车位重叠
  - 起始位置在道路区域内

```python
class ParkingMapNormal(object):
    def __init__(self, map_level=MAP_LEVEL, num_agents=NUM_AGENTS):
        self.num_agents = num_agents
        self.starts: List[State] = []      # 多个起始状态
        self.dests: List[State] = []       # 多个目标状态
        self.start_boxes: List[LinearRing] = []
        self.dest_boxes: List[LinearRing] = []
        # ... 保留原有的 xmin/xmax/ymin/ymax/obstacles
    
    def reset(self, case_id=None, path=None) -> List[State]:
        # 生成一个大停车场场景
        # 从中选取 num_agents 个不重叠的目标车位
        # 为每辆车生成不重叠的起始位置
        ...
```

**新增函数** `generate_multi_car_bay_parking(map_level, num_agents)`:
1. 生成一个足够大的停车场（一排 8-10 个车位）
2. 从中随机选取 `num_agents` 个非相邻车位作为目标
3. 其余车位由静态障碍车辆填充
4. 在道路区域为每辆车生成随机起始位置，确保互不重叠

---

### 3. 仿真环境（多智能体）

#### [MODIFY] [car_parking_base.py](file:///home/ake/1-500/HOPE_ge/src/env/car_parking_base.py)

**核心改动**：`CarParking` 从单车环境变为多车环境，管理 `NUM_AGENTS` 辆车。

**设计思路**：
- 维护 `self.vehicles: List[Vehicle]`（3 辆车）
- 每辆车有独立的观测（lidar + target）、独立的奖励
- 新增车间碰撞检测 & 车间安全距离计算
- 新增协同奖励（所有车都到达时给予额外奖励；最慢的车的时间惩罚）

```python
class MultiCarParking(gym.Env):
    def __init__(self, num_agents=NUM_AGENTS, ...):
        self.num_agents = num_agents
        self.vehicles: List[Vehicle] = []
        self.lidars: List[LidarSimlator] = []
        self.t_per_agent = [0.0] * num_agents  # 每车单独计时
        self.arrived = [False] * num_agents     # 每车到达状态
        ...
    
    def reset(self, ...):
        # 重置地图（生成多车位）
        # 为每辆车创建 Vehicle 对象并初始化
        ...
    
    def step(self, actions: List[np.ndarray]):
        # 逐车执行动作
        # 检测车间碰撞
        # 计算各车独立奖励 + 协同奖励
        # 返回 observations[], rewards[], statuses[], infos[]
        ...
```

**奖励函数改造**：

原有单车奖励保留为每辆车的独立奖励，新增两类多车协同奖励：

##### (a) 协同总时间奖励 `cooperative_time_reward`
```python
def _get_cooperative_time_reward(self):
    """
    鼓励所有车尽快同时完成泊车。
    - 当某车率先到达时，给予小奖励
    - 当所有车都到达时，根据最慢车的耗时给予大额奖励
    - 每一步对未完成的车施加时间惩罚
    """
    reward = 0
    if all(self.arrived):
        max_time = max(self.t_per_agent)
        reward = COOPERATIVE_TIME_WEIGHT * (1.0 - max_time / TOLERANT_TIME)
    return reward
```

##### (b) 智能体间安全性奖励 `inter_agent_safety_reward`
```python
def _get_safety_reward(self, agent_idx):
    """
    惩罚与其他车辆过近的行为，鼓励保持安全距离。
    - 计算当前车与所有其他车的最小距离
    - 距离 < SAFETY_DIST_THRESHOLD 时给予负奖励（线性惩罚）
    - 发生车间碰撞时给予大额负奖励
    """
    min_dist = float('inf')
    for j in range(self.num_agents):
        if j == agent_idx:
            continue
        dist = self.vehicles[agent_idx].box.distance(self.vehicles[j].box)
        min_dist = min(min_dist, dist)
    
    if min_dist < 0.01:  # 碰撞
        return -SAFETY_REWARD_WEIGHT * 10
    elif min_dist < SAFETY_DIST_THRESHOLD:
        return -SAFETY_REWARD_WEIGHT * (1 - min_dist / SAFETY_DIST_THRESHOLD)
    return 0
```

**渲染改造**：
- 同时渲染 3 辆车（不同颜色）及其目标车位
- 每辆车的目标车位用对应颜色绘制轮廓
- 可视化车间安全距离（当距离小于阈值时画红色警示线）

---

### 4. 环境包装器

#### [MODIFY] [env_wrapper.py](file:///home/ake/1-500/HOPE_ge/src/env/env_wrapper.py)

更新 `reward_shaping` 以处理多车奖励列表，聚合各车奖励为总奖励。

---

### 5. 可视化仪表盘

#### [NEW] [visualize_multi_car.py](file:///home/ake/1-500/HOPE_ge/src/evaluation/visualize_multi_car.py)

独立的可视化脚本，用 matplotlib 绘制关键指标：

1. **成功率**：每 N 个 episode 的滑动窗口成功率（全部到达 vs 部分到达 vs 失败）
2. **平均时间**：所有车完成泊车的平均时间 & 最慢车的平均时间
3. **安全性**：
   - 车间最小距离分布直方图
   - 车间碰撞次数随 episode 的变化
   - 安全距离违规比例

可视化场景界面：
- 使用 pygame 渲染 3 辆车同时泊车的动画
- 三个随机目标车位用不同颜色标注
- 三个随机起始位置用对应颜色标注
- 实时显示各车状态（距离目标、速度等）

---

## Open Questions

> [!IMPORTANT]
> **多智能体训练方式**：当前代码使用 PPO/SAC 单智能体训练。多车环境中，是希望：
> - (A) 所有车共享同一个策略网络（参数共享，self-play）？
> - (B) 每辆车独立训练各自的策略网络？
> - (C) 仅修改环境和可视化，训练部分暂不改动？
>
> **当前方案默认选择 (C)**——仅改造环境/奖励/可视化，不修改训练代码。环境输出兼容单智能体接口（可轮流控制每辆车）。

> [!IMPORTANT]
> **车辆是否可在目标车位停留不动**：当一辆车率先到达目标后，是否停止行动（freeze），还是继续受控直到所有车完成？
>
> **当前方案默认**：车辆到达后冻结不再执行动作。

---

## Verification Plan

### Automated Tests
1. 运行 `parking_map_normal.py` 的 `__main__` 测试，验证能生成 3 个不重叠的起始/目标位置
2. 实例化 `MultiCarParking` 并执行 `reset()` + 若干步 `step()`，确保无报错
3. 检查渲染窗口中 3 辆车都正确显示

### Manual Verification
1. 运行可视化脚本，目视确认：
   - 3 辆不同颜色的车出现在随机起始位置
   - 3 个对应颜色的目标车位在停车区域
   - 车辆不重叠
   - 指标图表正确显示
