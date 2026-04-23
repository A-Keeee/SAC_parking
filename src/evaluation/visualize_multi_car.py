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

from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.sac_agent import SACAgent as SAC
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import MultiCarParking
from env.env_wrapper import MultiCarParkingWrapper
from env.vehicle import VALID_SPEED, Status
from configs import *

def eval_multi_car(env, agent, episode=10, log_path='', post_proc_action=True):
    """
    函数核心功能：评估多智能体（多车）泊车模型的综合表现，并统计成功率、协同时间、安全距离等指标。
    
    关键变量：
    - num_agents: 当前环境中的车辆总数（动态从环境配置中获取）。
    - success_counts: 记录每个回合成功泊车的车辆数量的列表。
    - coop_times: 记录所有成功车辆完成泊车所需最大步数的列表。
    - safety_dists: 记录每个回合中车辆之间最小安全距离的列表。
    
    重要逻辑步骤：
    1. 遍历指定的评估回合数 (episode)，重置环境和智能体状态。
    2. 在每个回合中循环执行环境步进，直到所有车辆完成（到达、碰撞、超时或越界）。
    3. 获取每个智能体的动作，如果是后处理模式 (post_proc_action)，则调用 choose_action。
    4. 实时计算当前帧任意两车之间的欧氏距离，更新本回合最小距离 (min_dist_episode)。
    5. 回合结束后，统计成功车辆数、协同耗时，并将图表数据保存到日志目录。
    """
    num_agents = env.env.num_agents
    
    # Metrics
    success_counts = [] # Number of cars successfully parked per episode
    coop_times = []     # Total time until all successfully parked cars finish
    safety_dists = []   # Minimum distance recorded per episode
    
    for i in trange(episode):
        obs_list = env.reset(i+1)
        agent.reset()
        done_list = [False] * num_agents
        
        step_num = 0
        min_dist_episode = float('inf')
        
        while not all(done_list):
            step_num += 1
            
            # Select actions
            actions = []
            for j in range(num_agents):
                if not done_list[j]:
                    if post_proc_action:
                        action, _ = agent.choose_action(obs_list[j])
                    else:
                        action, _ = agent.get_action(obs_list[j])
                    actions.append(action)
                else:
                    # Dummy action if already done (won't be applied by env anyway)
                    actions.append(env.action_space.sample())
            
            next_obs_list, reward_list, done_list_new, info_list = env.step(actions)
            
            # Record safety dist
            # 计算并记录本回合多车之间的最小物理距离
            for j in range(num_agents):
                for k in range(j+1, num_agents):
                    dist = env.env.vehicles[j].box.distance(env.env.vehicles[k].box)
                    min_dist_episode = min(min_dist_episode, dist)
            
            obs_list = next_obs_list
            done_list = done_list_new

        # Evaluate outcome
        statuses = [info['status'] for info in info_list]
        num_success = sum([1 for s in statuses if s == Status.ARRIVED])
        success_counts.append(num_success)
        
        if num_success > 0:
            # Time of the last successful car
            max_time = max([env.env.t_per_agent[j] for j in range(num_agents) if statuses[j] == Status.ARRIVED])
            coop_times.append(max_time)
            
        safety_dists.append(min_dist_episode)

    print('#'*15)
    print('MULTI-CAR EVALUATION RESULT:')
    print(f'Total episodes: {episode}')
    if episode > 0:
        print(f'All {num_agents} cars arrived success rate: {success_counts.count(num_agents)/episode*100:.2f}%')
        print(f'At least 1 car arrived success rate: {(episode - success_counts.count(0))/episode*100:.2f}%')
    else:
        print(f'All {num_agents} cars arrived success rate: 0.00%')
        print(f'At least 1 car arrived success rate: 0.00%')
    
    if len(coop_times) > 0:
        print(f'Average cooperative time (successful episodes): {np.mean(coop_times):.2f} steps')
    if len(safety_dists) > 0 and num_agents > 1:
        print(f'Average minimum safety distance: {np.mean(safety_dists):.2f} m')
    
    # Save plots
    if log_path:
        # 1. Success Rate Distribution
        plt.figure(figsize=(8, 6))
        counts = [success_counts.count(j) for j in range(num_agents+1)]
        plt.bar(range(num_agents+1), counts, color='skyblue')
        plt.xlabel('Number of Cars Successfully Parked')
        plt.ylabel('Episode Count')
        plt.title('Parking Success Distribution')
        plt.xticks(range(num_agents+1))
        plt.savefig(os.path.join(log_path, 'success_distribution.png'))
        plt.close()
        
        # 2. Average Time
        if len(coop_times) > 0:
            plt.figure(figsize=(8, 6))
            plt.hist(coop_times, bins=20, color='lightgreen')
            plt.xlabel('Cooperative Parking Time (steps)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Cooperative Parking Time')
            plt.savefig(os.path.join(log_path, 'cooperative_time.png'))
            plt.close()
            
        # 3. Safety Distance (仅在多于一辆车时绘制该图表)
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

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--eval_episode', type=int, default=10)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=True)
    args = parser.parse_args()

    checkpoint_path = args.ckpt_path
    print('ckpt path: ',checkpoint_path)
    verbose = args.verbose

    # 【修复核心】: 将原本硬编码的 num_agents=3 替换为配置文件中的 NUM_AGENTS 变量
    if args.visualize:
        raw_env = MultiCarParking(fps=100, verbose=verbose, num_agents=NUM_AGENTS)
    else:
        raw_env = MultiCarParking(fps=100, verbose=verbose, render_mode='rgb_array', num_agents=NUM_AGENTS)
        
    env = MultiCarParkingWrapper(raw_env)

    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = os.path.join(os.path.dirname(__file__), '..', 'log', 'eval_multi', timestamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if checkpoint_path is None:
        Agent_type = SAC # Default
    else:
        Agent_type = PPO if 'ppo' in checkpoint_path.lower() else SAC

    seed = SEED
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    step_ratio = env.env.vehicles[0].kinetic_model.step_len*env.env.vehicles[0].kinetic_model.n_step*VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)

    eval_episode = args.eval_episode
    choose_action = True if isinstance(rl_agent, PPO) else False
    
    with torch.no_grad():
        # eval on normalize
        env.env.set_level('Normal')
        eval_multi_car(env, parking_agent, episode=eval_episode, log_path=save_path, post_proc_action=choose_action)

    env.close()