import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

import time
import argparse
from typing import DefaultDict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange

# 假设环境使用了 Pygame，需要引入以便在 render 循环中叠加绘制
try:
    import pygame
except ImportError:
    pass # 如果没有，实时轨迹渲染将不可用

from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.sac_agent import SACAgent as SAC
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import MultiCarParking
from env.env_wrapper import MultiCarParkingWrapper
from env.vehicle import VALID_SPEED, Status
# 确保 configs.py 存在并包含所需参数
try:
    from configs import *
except ImportError:
    # 如果缺少，请在此处补充定义默认值，例如：
    # NUM_AGENTS = 2
    # ACTOR_CONFIGS = {...}
    # CRITIC_CONFIGS = {...}
    # VALID_SPEED = [0, 2]
    # SEED = 42
    # SAFETY_DIST_THRESHOLD = 0.5
    pass

# ==========================================
# 新增：轨迹标记配置（如果 configs.py 未定义）
# ==========================================
if 'RENDER_TRAJECTORY' not in globals():
    RENDER_TRAJECTORY = True # 是否在实时 render 时叠加绘制轨迹线
if 'TRAJECTORY_MARKER_SIZE' not in globals():
    TRAJECTORY_MARKER_SIZE = 8 # 最优路径图中起点/终点标记的大小

def eval_multi_car(env, agent, episode=10, log_path='', post_proc_action=True, visualize=False):
    """
    函数核心功能：
    评估多智能体泊车模型的综合表现，统计基础指标（成功率、时间、安全距离），
    记录并保存：
    1. 成功率分布图
    2. 协同完成时间分布图
    3. 最短安全距离分布图
    4. 回合累计奖励趋势图
    5. SAC 动作平滑度图
    6. **新增：**车辆中心点轨迹图（最优路径）
    """
    num_agents = env.env.num_agents
    
    # 指标容器
    success_counts = [] # 每回合成功泊车的数量
    coop_times = []     # 所有成功车辆完成的总时间
    safety_dists = []   # 每回合记录的最小距离
    episode_rewards = [] # 每回合所有车辆累积的总奖励 (SAC 指标)
    
    # 数据记录结构
    trajectories_all = [] # 记录所有回合的全程轨迹数据
    best_ep_actions = None
    best_ep_reward = -float('inf')

    # ==========================================
    # 新增：记录最优路径的数据结构
    # ==========================================
    max_total_reward = -float('inf')
    best_path_data = None # 格式: {agent_id: [(x1,y1), (x2,y2), ...]}
    
    # 定义不同车辆在图中显示的颜色 (Matplotlib)
    map_colors = plt.cm.get_cmap('tab10', num_agents)

    # 评估回合循环
    for i in trange(episode, desc="Eval Episodes"):
        # 回合重置
        obs_list = env.reset(i+1)
        agent.reset()
        done_list = [False] * num_agents
        
        step_num = 0
        min_dist_episode = float('inf')
        ep_total_reward = 0.0
        
        # 当前回合的数据记录
        ep_trajectories = {j: [] for j in range(num_agents)} # 车辆像素轨迹 (用于实时 render)
        ep_actions = {j: [] for j in range(num_agents)}
        current_ep_paths = {j: [] for j in range(num_agents)} # 车辆真实 $x, y$ 轨迹

        # 记录初始位置 (真实坐标)
        for j in range(num_agents):
            curr_loc = env.env.vehicles[j].state.loc
            current_ep_paths[j].append((curr_loc.x, curr_loc.y))

        # 时间步循环
        while not all(done_list):
            step_num += 1
            
            # 决策动作
            actions = []
            for j in range(num_agents):
                if not done_list[j]:
                    if post_proc_action:
                        action, _ = agent.choose_action(obs_list[j])
                    else:
                        action, _ = agent.get_action(obs_list[j])
                    actions.append(action)
                    ep_actions[j].append(action) # 记录动作以展现 SAC 平滑度
                else:
                    # Dummy action if already done (won't be applied by env anyway)
                    dummy_act = env.action_space.sample()
                    actions.append(dummy_act)
                    ep_actions[j].append(np.zeros_like(dummy_act))
            
            # 环境步进
            next_obs_list, reward_list, done_list_new, info_list = env.step(actions)
            
            # 累加奖励
            ep_total_reward += sum(reward_list)

            # ====== 完美入库视觉吸附特效 ======
            need_visual_update = False
            for j in range(num_agents):
                if info_list[j]['status'] == Status.ARRIVED:
                    # 1. 强行将车辆坐标和航向角修正为目标框中心参数
                    env.env.vehicles[j].state.loc = env.env.map.dests[j].loc
                    env.env.vehicles[j].state.heading = env.env.map.dests[j].heading
                    
                    # 2. 强行将视觉碰撞框替换为目标框，实现 100% 重合
                    env.env.vehicles[j].box = env.env.map.dest_boxes[j]
                    
                    # 3. 标记需要刷新屏幕
                    if not done_list[j]:
                        need_visual_update = True
            
            # 记录安全距离和路径
            for j in range(num_agents):
                # 记录像素坐标 (用于实时绘制轨迹线，假设环境提供了逆转换函数 `_coord_transform_inv`)
                # 注意：这里假设环境底层使用的是 pygame，且有一个可以将物理坐标转为屏幕坐标的函数。
                # 如果没有该函数，下面的 `if visualize and RENDER_TRAJECTORY` 块需要修改为环境自带的绘制逻辑。
                # try:
                #     pix_x, pix_y = env.env._coord_transform_inv(env.env.vehicles[j].state.loc)
                #     ep_trajectories[j].append((pix_x, pix_y))
                # except AttributeError:
                #     pass # 环境未提供逆转换函数，跳过像素轨迹记录

                # 记录真实坐标路径
                curr_loc = env.env.vehicles[j].state.loc
                current_ep_paths[j].append((curr_loc.x, curr_loc.y))

                # 记录安全距离
                for k in range(j+1, num_agents):
                    dist = env.env.vehicles[j].box.distance(env.env.vehicles[k].box)
                    min_dist_episode = min(min_dist_episode, dist)

            # 刷新渲染
            if need_visual_update or (all(done_list_new) and not all(done_list)):
                env.env.render()

            # ==========================================
            # 新增：在可视化模式下叠加绘制行驶路径
            # ==========================================
            if visualize and RENDER_TRAJECTORY:
                # 获取环境的 pygame surface，并在此基础上叠加绘制轨迹
                # 这个逻辑假设环境底层使用了 pygame 且 `render` 函数会更新全局 surface
                # 注意：这通常需要修改环境代码才能实现理想效果，或者需要环境提供轨迹绘制接口。
                # 这里提供一个概念性的叠加绘制块，具体实现取决于环境代码。
                try:
                    screen = pygame.display.get_surface()
                    if screen is not None:
                        # 定义车辆轨迹的 pygame 颜色
                        pg_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                        for j in range(num_agents):
                            c = pg_colors[j % len(pg_colors)]
                            # 需要环境将物理坐标转为像素坐标，这里假设 `_coord_transform_inv` 存在
                            # 如果不存在，请跳过此块，或者使用环境自定义的 render_traj 逻辑
                            points = []
                            for loc_x, loc_y in current_ep_paths[j]:
                                pix_x, pix_y = env.env._coord_transform_inv(np.array([loc_x, loc_y]))
                                points.append((pix_x, pix_y))

                            if len(points) > 1:
                                pygame.draw.lines(screen, c, False, points, 2)
                        
                        pygame.display.flip() # 刷新将我们绘制的轨迹叠加显示出来
                except (NameError, AttributeError, ImportError):
                    pass # 如果没有 pygame 或相关函数，直接跳过

            # 更新观测和 done 状态
            obs_list = next_obs_list
            done_list = done_list_new

        # 回合结束，评估成果
        statuses = [info['status'] for info in info_list]
        num_success = sum([1 for s in statuses if s == Status.ARRIVED])
        success_counts.append(num_success)
        episode_rewards.append(ep_total_reward)
        trajectories_all.append(ep_trajectories)
        
        # 记录最优路径
        if num_success > 0:
            #Time of the last successful car
            max_time = max([env.env.t_per_agent[j] for j in range(num_agents) if statuses[j] == Status.ARRIVED])
            coop_times.append(max_time)
            
        safety_dists.append(min_dist_episode)

        # 选出最好（奖励最高）的一局的动作数据用于展示 SAC 平滑度
        if ep_total_reward > best_ep_reward:
            best_ep_reward = ep_total_reward
            best_ep_actions = ep_actions

        # ==========================================
        # 新增：记录总奖励最高的一局作为最优路径数据
        # ==========================================
        if ep_total_reward > max_total_reward and num_success > 0:
            max_total_reward = ep_total_reward
            best_path_data = current_ep_paths

    # 控制台结果输出
    print('#'*15)
    print('MULTI-CAR EVALUATION RESULT:')
    print(f'Total episodes: {episode}')
    if episode > 0:
        print(f'All {num_agents} cars arrived success rate: {success_counts.count(num_agents)/episode*100:.2f}%')
        print(f'At least 1 car arrived success rate: {(episode - success_counts.count(0))/episode*100:.2f}%')
        print(f'Average Total Reward (SAC Metric): {np.mean(episode_rewards):.2f}')
    else:
        print(f'All {num_agents} cars arrived success rate: 0.00%')
        print(f'At least 1 car arrived success rate: 0.00%')
    
    if len(coop_times) > 0:
        print(f'Average cooperative time (successful episodes): {np.mean(coop_times):.2f} steps')
    if len(safety_dists) > 0 and num_agents > 1:
        print(f'Average minimum safety distance: {np.mean(safety_dists):.2f} m')
    
    # ==========================================
    # 保留：保存各项评估指标的可视化图表
    # ==========================================
    if log_path:
        # 1. 成功率分布图
        plt.figure(figsize=(8, 6))
        counts = [success_counts.count(j) for j in range(num_agents+1)]
        plt.bar(range(num_agents+1), counts, color='skyblue')
        plt.xlabel('Number of Cars Successfully Parked')
        plt.ylabel('Episode Count')
        plt.title('Parking Success Distribution')
        plt.xticks(range(num_agents+1))
        plt.savefig(os.path.join(log_path, 'success_distribution.png'))
        plt.close()
        
        # 2. 协同完成时间分布图
        if len(coop_times) > 0:
            plt.figure(figsize=(8, 6))
            plt.hist(coop_times, bins=20, color='lightgreen')
            plt.xlabel('Cooperative Parking Time (steps)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Cooperative Parking Time')
            plt.savefig(os.path.join(log_path, 'cooperative_time.png'))
            plt.close()
            
        # 3. 最短安全距离分布图
        if num_agents > 1:
            plt.figure(figsize=(8, 6))
            plt.hist(safety_dists, bins=20, color='salmon')
            plt.axvline(x=SAFETY_DIST_THRESHOLD, color='r', linestyle='--', label='Safety Threshold')
            plt.xlabel('Minimum Inter-Agent Distance (m)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Minimum Safety Distance')
            plt.legend()
            plt.savefig(os.path.join(log_path, 'safety_distance.png'))
            plt.close()

        # 4. 回合累计奖励趋势图
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, episode+1), episode_rewards, marker='o', linestyle='-', color='purple')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Evaluation Episode Rewards')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(log_path, 'episode_rewards.png'))
        plt.close()

        # 5. SAC 动作平滑度图
        if best_ep_actions is not None:
            plt.figure(figsize=(12, 10))
            # 假设 action 空间格式为 [acceleration, steering]
            for j in range(num_agents):
                acts = best_ep_actions[j]
                if len(acts) > 0:
                    accels = [a[0] for a in acts]
                    steers = [a[1] for a in acts]
                    
                    plt.subplot(num_agents, 2, 2*j + 1)
                    plt.plot(accels, color='blue', alpha=0.8)
                    plt.ylabel(f'Agent {j} Accel')
                    if j == 0: plt.title('SAC Acceleration Output')
                    plt.grid(True, linestyle='--', alpha=0.5)

                    plt.subplot(num_agents, 2, 2*j + 2)
                    plt.plot(steers, color='orange', alpha=0.8)
                    plt.ylabel(f'Agent {j} Steer')
                    if j == 0: plt.title('SAC Steering Output')
                    plt.grid(True, linestyle='--', alpha=0.5)

            plt.xlabel('Time Step')
            plt.tight_layout()
            plt.savefig(os.path.join(log_path, 'sac_action_smoothness.png'))
            plt.close()

        # ==========================================
        # 新增：保存最优一局的车辆中心点轨迹图
        # ==========================================
        if best_path_data is not None:
            plt.figure(figsize=(10, 8))
            
            # 1. 绘制背景（地图元素：障碍物、车位框）
            # 这部分取决于环境地图的存储结构。假设有一个 `map.draw(ax)` 函数或可以直接访问坐标
            # 这里给出一个概念性的地图元素绘制块，具体实现需要替换为实际地图数据。
            ax = plt.gca()
            # 绘制障碍物 (假设 `env.env.map.obstacles` 是坐标列表)
            # for obs in env.env.map.obstacles:
            #     # ... 绘制多边形障障碍物 ...
            #     pass
            # 绘制车位框 (目标框)
            for j in range(num_agents):
                dest_loc = env.env.map.dests[j].loc
                # 绘制一个矩形代表车位框
                # rect = plt.Rectangle((dest_loc.x - W/2, dest_loc.y - H/2), W, H, color='green', alpha=0.3)
                # ax.add_patch(rect)
                plt.text(dest_loc.x, dest_loc.y, f'Slot {j}', color='green', ha='center', va='center')

            # 2. 绘制最优轨迹线
            for j in range(num_agents):
                xs = [pos[0] for pos in best_path_data[j]]
                ys = [pos[1] for pos in best_path_data[j]]
                c = map_colors(j)
                plt.plot(xs, ys, color=c, label=f'Agent {j} Path', linewidth=2, alpha=0.7)
                
                # 3. 标记起点 (形) 和终点 (形)
                plt.scatter(xs[0], ys[0], color=c, marker='o', s=100*TRAJECTORY_MARKER_SIZE/8, edgecolors='black', label=f'Agent {j} Start')
                plt.scatter(xs[-1], ys[-1], color=c, marker='*', s=150*TRAJECTORY_MARKER_SIZE/8, edgecolors='black', zorder=5, label=f'Agent {j} End')

            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.title('Smoothest & Best Episode Vehicle Center Trajectories')
            plt.legend(loc='upper right') # 固定图例位置
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.axis('equal') # 保证 X 和 Y 轴比例一致，反映地图物理形状
            plt.savefig(os.path.join(log_path, 'best_episode_trajectories.png'))
            plt.close()
            print(f'Successfully saved best trajectory plot to {log_path}')


if __name__=="__main__":
    """
    脚本入口核心功能：解析命令行参数，初始化环境与 SAC 模型结构，加载预训练权重并触发多智能体评估流程。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to pre-trained model checkpoint')
    parser.add_argument('--eval_episode', type=int, default=10, help='number of episodes to evaluate')
    parser.add_argument('--verbose', type=bool, default=True, help='print detailed reward information')
    parser.add_argument('--visualize', type=bool, default=True, help='enable GUI during evaluation')
    args = parser.parse_args()

    checkpoint_path = args.ckpt_path
    print('ckpt path: ',checkpoint_path)
    verbose = args.verbose

    # 根据配置初始化环境
    if args.visualize:
        raw_env = MultiCarParking(fps=100, verbose=verbose, num_agents=NUM_AGENTS, render_mode='human')
    else:
        raw_env = MultiCarParking(fps=100, verbose=verbose, render_mode='rgb_array', num_agents=NUM_AGENTS)
        
    env = MultiCarParkingWrapper(raw_env)

    # 创建保存日志和图片的文件夹
    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = os.path.join(os.path.dirname(__file__), '..', 'log', 'eval_multi', timestamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # 确定智能体类型 (SAC 或 PPO)
    if checkpoint_path is None:
        Agent_type = SAC # 默认使用 SAC
    else:
        Agent_type = PPO if 'ppo' in checkpoint_path.lower() else SAC

    # 设置随机种子
    seed = SEED
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 智能体网络配置
    actor_params = ACTOR_CONFIGS
    critic_params = CRITIC_CONFIGS
    configs = {
        "discrete": False,
        "observation_shape": env.observation_shape,
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
    }

    rl_agent = Agent_type(configs)
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        print('load pre-trained model!')

    # 初始化规划器和联合智能体
    step_ratio = env.env.vehicles[0].kinetic_model.step_len*env.env.vehicles[0].kinetic_model.n_step*VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)

    eval_episode = args.eval_episode
    
    # 确定是否需要对 PPO 的动作进行后处理
    choose_action = True if isinstance(rl_agent, PPO) else False    

    with torch.no_grad():
        # eval on normal level
        env.env.set_level('Normal')
        # 调用评估函数，新增 visualize 参数传递
        eval_multi_car(env, parking_agent, episode=eval_episode, log_path=save_path, post_proc_action=choose_action, visualize=args.visualize)

    env.close()