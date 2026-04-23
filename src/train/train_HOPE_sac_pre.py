import sys
sys.path.append("..")
sys.path.append(".")
import time
import os
from shutil import copyfile
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from model.agent.sac_agent import SACAgent as SAC
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking, MultiCarParking
from env.env_wrapper import CarParkingWrapper, MultiCarParkingWrapper
from env.vehicle import VALID_SPEED, Status
from evaluation.eval_utils import eval
from evaluation.visualize_multi_car import eval_multi_car
from configs import *


class SceneChoose():
    def __init__(self) -> None:
        self.scene_types = {0: 'Normal', 1: 'Complex', 2: 'Extrem'}
        self.target_success_rate = np.array([0.95, 0.95, 0.9])
        self.success_record = {}
        for scene_name in self.scene_types:
            self.success_record[scene_name] = []
        self.scene_record =[]
        self.history_horizon = 200
        
    def choose_case(self,):
        if len(self.scene_record) < self.history_horizon:
            scene_chosen = self._choose_case_uniform()
        else:
            if np.random.random() > 0.5:
                scene_chosen = self._choose_case_worst_perform()
            else:
                scene_chosen = self._choose_case_uniform()
        self.scene_record.append(scene_chosen)
        return self.scene_types[scene_chosen]
    
    def update_success_record(self, success:int):
        self.success_record[self.scene_record[-1]].append(success)

    def _choose_case_uniform(self,):
        case_count = np.zeros(len(self.scene_types))
        for i in range(min(len(self.scene_record), self.history_horizon)):
            scene_id = self.scene_record[-(i+1)]
            case_count[scene_id] += 1
        return np.argmin(case_count)
    
    def _choose_case_worst_perform(self,):
        success_rate =[]
        for i in self.success_record.keys():
            idx = int(i)
            recent_success_record = self.success_record[idx][-min(250, len(self.success_record[idx])):]
            success_rate.append(np.sum(recent_success_record)/len(recent_success_record))
        fail_rate = self.target_success_rate - np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.01, 1)
        fail_rate = fail_rate/np.sum(fail_rate)
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)


class DlpCaseChoose():
    def __init__(self) -> None:
        self.dlp_case_num = 248
        self.case_record =[]
        self.case_success_rate = {}
        for i in range(self.dlp_case_num):
            self.case_success_rate[str(i)] =[]
        self.horizon = 500
    
    def choose_case(self,):
        if np.random.random()<0.2 or len(self.case_record)<self.horizon:
            return np.random.randint(0, self.dlp_case_num)
        success_rate =[]
        for i in range(self.dlp_case_num):
            idx = str(i)
            if len(self.case_success_rate[idx]) <= 1:
                success_rate.append(0)
            else:
                recent_success_record = self.case_success_rate[idx][-min(10, len(self.case_success_rate[idx])):]
                success_rate.append(np.sum(recent_success_record)/len(recent_success_record))
        fail_rate = 1-np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.005, 1)
        fail_rate = fail_rate/np.sum(fail_rate)
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)
    
    def update_success_record(self, success:int, case_id:int):
        self.case_success_rate[str(case_id)].append(success)
        self.case_record.append(case_id)


# 修复 boolean arguments 解析问题
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ckpt', type=str, default=None) # './model/ckpt/SAC.pt'
    parser.add_argument('--img_ckpt', type=str, default='./model/ckpt/autoencoder.pt')
    parser.add_argument('--train_episode', type=int, default=100000)
    parser.add_argument('--eval_episode', type=int, default=200)
    parser.add_argument('--verbose', type=str2bool, default=True)
    parser.add_argument('--visualize', type=str2bool, default=False)
    args = parser.parse_args()

    verbose = args.verbose

    if args.visualize:
        raw_env = MultiCarParking(fps=100, verbose=verbose, num_agents=NUM_AGENTS)
    else:
        raw_env = MultiCarParking(fps=100, verbose=verbose, render_mode='rgb_array', num_agents=NUM_AGENTS)
    
    env = MultiCarParkingWrapper(raw_env)
    scene_chooser = SceneChoose()
    dlp_case_chooser = DlpCaseChoose()

    # the path to log and save model
    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/exp/sac_multi_%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)
    
    # configs log
    copyfile('./configs.py', save_path+'configs.txt')
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)

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
    print('observation_space:',env.observation_space)

    rl_agent = SAC(configs)
    
    # 修复加载逻辑，加强路径验证且设置 params_only=True
    checkpoint_path = args.agent_ckpt
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            # 将 params_only 设为 True 能够一并加载优化器和 SAC 关键的探索度参数 alpha
            rl_agent.load(checkpoint_path, params_only=True)
            print(f'Successfully loaded pre-trained model from {checkpoint_path}!')
        else:
            print(f'Error: Checkpoint {checkpoint_path} not found! Exiting...')
            sys.exit(1)

    img_encoder_checkpoint =  args.img_ckpt if USE_IMG else None
    if img_encoder_checkpoint is not None and os.path.exists(img_encoder_checkpoint):
        # 修复特征空间被重置的问题：如果已经加载了完整的SAC预训练模型，则不要再用基础的AE覆盖它
        if checkpoint_path is None:
            rl_agent.load_img_encoder(img_encoder_checkpoint, require_grad=UPDATE_IMG_ENCODE)
        else:
            print(f'Skipped loading basic img encoder {img_encoder_checkpoint} to avoid overwriting the pre-trained SAC actor.')

    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)

    reward_list =[]
    reward_per_state_list = []
    reward_info_list =[]
    case_id_list = []
    succ_record =[]
    total_step_num = 0
    best_success_rate = [0, 0, 0, 0]

    for i in range(args.train_episode):
        scene_chosen = scene_chooser.choose_case()
        if scene_chosen == 'dlp':
            case_id = dlp_case_chooser.choose_case()
        else:
            case_id = None
            
        # obs_list contains observations for all N cars
        obs_list = env.reset(case_id, None, scene_chosen)
        parking_agent.reset()
        case_id_list.append(env.map.case_id)
        
        done_list = [False] * env.num_agents
        total_reward = [0.0] * env.num_agents
        step_num = 0
        reward_info =[]
        
        while not all(done_list):
            step_num += 1
            total_step_num += 1
            
            actions = []
            log_probs =[]
            for agent_idx in range(env.num_agents):
                # 修复核心 Bug: 如果提供了预训练权重 (checkpoint_path 不为空)，则跳过前期的全随机探索
                if total_step_num <= parking_agent.configs.memory_size and checkpoint_path is None:
                    action = env.action_space.sample()
                    log_prob = parking_agent.get_log_prob(obs_list[agent_idx], action)
                else:
                    action, log_prob = parking_agent.get_action(obs_list[agent_idx])
                
                actions.append(action)
                log_probs.append(log_prob)

            next_obs_list, reward_list_step, done_list_step, info_list = env.step(actions)
            reward_info.append(list(info_list[0]['reward_info'].values())) # Track first agent for summary
            
            for agent_idx in range(env.num_agents):
                if not done_list[agent_idx]:
                    # Only collect transitions for active agents
                    total_reward[agent_idx] += reward_list_step[agent_idx]
                    reward_per_state_list.append(reward_list_step[agent_idx])
                    parking_agent.push_memory((
                        obs_list[agent_idx], 
                        actions[agent_idx], 
                        reward_list_step[agent_idx], 
                        done_list_step[agent_idx], 
                        log_probs[agent_idx], 
                        next_obs_list[agent_idx]
                    ))
            
            obs_list = next_obs_list
            done_list = done_list_step
            
            # Train the centralized model
            if total_step_num > parking_agent.configs.memory_size and total_step_num % 10 == 0:
                actor_loss, critic_loss = parking_agent.update()
                if total_step_num % 200 == 0:
                    writer.add_scalar("actor_loss", actor_loss, i)
                    writer.add_scalar("critic_loss", critic_loss, i)

            if all(done_list):
                # Only when all agents finish, consider episode successful if ALL arrived
                # Or count how many arrived
                all_arrived = True
                for agent_idx in range(env.num_agents):
                    if info_list[agent_idx]['status'] != Status.ARRIVED:
                        all_arrived = False
                
                if all_arrived:
                    succ_record.append(1)
                    scene_chooser.update_success_record(1)
                    if scene_chosen == 'dlp':
                        dlp_case_chooser.update_success_record(1, case_id)
                else:
                    succ_record.append(0)
                    scene_chooser.update_success_record(0)
                    if scene_chosen == 'dlp':
                        dlp_case_chooser.update_success_record(0, case_id)

            
        writer.add_scalar("total_reward", np.mean(total_reward), i)
        writer.add_scalar("avg_reward", np.mean(reward_per_state_list[-1000:]), i)
        writer.add_scalar("action_std0", parking_agent.log_std.detach().cpu().numpy().reshape(-1)[0],i)
        writer.add_scalar("action_std1", parking_agent.log_std.detach().cpu().numpy().reshape(-1)[1],i)
        writer.add_scalar("alpha", parking_agent.alpha.detach().cpu().numpy().reshape(-1)[0],i)
        for type_id in scene_chooser.scene_types:
            writer.add_scalar("success_rate_%s"%scene_chooser.scene_types[type_id],
                np.mean(scene_chooser.success_record[type_id][-100:]), i)
        writer.add_scalar("step_num", step_num, i)
        
        reward_list.append(np.mean(total_reward))
        reward_info = np.sum(np.array(reward_info), axis=0)
        reward_info = np.round(reward_info,2)
        reward_info_list.append(list(reward_info))

        if verbose and i%10==0 and i>0:
            print('success rate:',np.sum(succ_record),'/',len(succ_record))
            print(parking_agent.log_std.detach().cpu().numpy().reshape(-1), parking_agent.alpha.detach().cpu().numpy().reshape(-1))
            print("episode:%s  average reward:%s"%(i,np.mean(reward_list[-50:])))
            print(np.mean(parking_agent.actor_loss_list[-100:]),np.mean(parking_agent.critic_loss_list[-100:]))
            print('time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward')
            for j in range(10):
                print(case_id_list[-(10-j)],reward_list[-(10-j)],reward_info_list[-(10-j)])
            print("")

        # save best model
        for type_id in scene_chooser.scene_types:
            success_rate_normal = np.mean(scene_chooser.success_record[0][-100:])
            success_rate_complex = np.mean(scene_chooser.success_record[1][-100:])
            success_rate_extreme = np.mean(scene_chooser.success_record[2][-100:])
        if success_rate_normal >= best_success_rate[0] and success_rate_complex >= best_success_rate[1] and\
            success_rate_extreme >= best_success_rate[2] and i>100:
            raw_best_success_rate = np.array([success_rate_normal, success_rate_complex, success_rate_extreme])
            best_success_rate = list(np.minimum(raw_best_success_rate, scene_chooser.target_success_rate))
            parking_agent.save("%s/SAC_multi_best.pt" % (save_path),params_only=True)
            f_best_log = open(save_path+'best.txt', 'w')
            f_best_log.write('epoch: %s, success rate: %s %s %s'%(i+1, raw_best_success_rate[0],
                                raw_best_success_rate[1], raw_best_success_rate[2]))
            f_best_log.close()

        if (i+1) % 2000 == 0:
            parking_agent.save("%s/SAC_multi_%s.pt" % (save_path, i),params_only=True)
        

        if verbose and i%20==0:
            episodes =[j for j in range(len(reward_list))]
            mean_reward = [np.mean(reward_list[max(0,j-50):j+1]) for j in range(len(reward_list))]
            plt.plot(episodes,reward_list)
            plt.plot(episodes,mean_reward)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            f = plt.gcf()
            f.savefig('%s/reward.png'%save_path)
            f.clear()

    eval_episode = args.eval_episode
    choose_action = False
    with torch.no_grad():
        # eval on extreme
        env.set_level('Extrem')
        log_path = save_path+'/extreme'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval_multi_car(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        
        # eval on complex
        env.set_level('Complex')
        log_path = save_path+'/complex'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval_multi_car(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        
        # eval on normalize
        env.set_level('Normal')
        log_path = save_path+'/normalize'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval_multi_car(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)

    env.close()