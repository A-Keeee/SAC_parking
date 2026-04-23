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
            
            # ====== 新增：完美入库视觉吸附特效 ======
            need_visual_update = False
            for j in range(num_agents):
                if info_list[j]['status'] == Status.ARRIVED:
                    # 1. 强行将车辆的坐标与航向角修正为目标车位的中心参数
                    # 修复：直接替换 loc 对象，而不是修改只读的 x 和 y 属性
                    env.env.vehicles[j].state.loc = env.env.map.dests[j].loc
                    env.env.vehicles[j].state.heading = env.env.map.dests[j].heading
                    
                    # 2. 强行将渲染用的物理碰撞框替换为目标框，实现 100% 视觉重合
                    env.env.vehicles[j].box = env.env.map.dest_boxes[j]
                    
                    # 3. 如果是刚刚在这一步到达的，标记需要刷新屏幕
                    if not done_list[j]:
                        need_visual_update = True
                        
            if need_visual_update or (all(done_list_new) and not all(done_list)):
                env.env.render()

            # Record safety dist
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
            
        # 3. Safety Distance
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