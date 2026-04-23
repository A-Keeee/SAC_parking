# 多车协同泊车任务跟踪

- [ ] 1. 更新 `configs.py`
  - [ ] 添加多车配置项 (NUM_AGENTS, COLOR_POOL 等)
  - [ ] 修改窗口大小

- [ ] 2. 修改 `parking_map_normal.py`
  - [ ] 增加 `NUM_AGENTS` 支持
  - [ ] 实现生成 3 个不重叠的起点的逻辑
  - [ ] 实现生成 3 个不相邻的目标车位的逻辑
  - [ ] 更新 `ParkingMapNormal.reset()` 以支持多车初始化

- [ ] 3. 修改 `car_parking_base.py`
  - [ ] 初始化 `NUM_AGENTS` 辆车和雷达
  - [ ] 更新 `step()` 逻辑，支持多车独立步进
  - [ ] 增加车间安全距离和碰撞检测
  - [ ] 实现协同时间奖励 (`_get_cooperative_time_reward`)
  - [ ] 实现智能体间安全性奖励 (`_get_safety_reward`)
  - [ ] 更新渲染逻辑，绘制 3 辆车及其轨迹和目标车位

- [ ] 4. 修改 `env_wrapper.py`
  - [ ] 更新 `reward_shaping` 处理多个奖励

- [ ] 5. 创建 `evaluation/visualize_multi_car.py`
  - [ ] 编写测试脚本运行多车环境
  - [ ] 实现性能指标统计 (成功率，平均时间，安全性) 
  - [ ] 实时绘制统计图表

- [ ] 6. 测试与验证
  - [ ] 验证地图生成无重叠
  - [ ] 验证环境可以正常运行 `step()`
  - [ ] 验证渲染正常
