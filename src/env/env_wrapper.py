import numpy as np
from gym import Wrapper

from env.car_parking_base import CarParking, MultiCarParking
from env.vehicle import Status
from configs import REWARD_WEIGHT, REWARD_RATIO

def reward_shaping(*args):
    '''
    函数核心功能：单车环境的奖励塑形函数。
    '''
    obs, reward_info, status, info = args
    if status == Status.CONTINUE:
        reward = 0
        for reward_type in REWARD_WEIGHT.keys():
            reward += REWARD_WEIGHT[reward_type]*reward_info[reward_type]
    elif status == Status.OUTBOUND:
        reward = -50
    elif status == Status.OUTTIME:
        reward = -1
    elif status == Status.ARRIVED:
        reward = 50
    elif status == Status.COLLIDED:
        reward = -50
    else:
        print(status)
        print('Never reach here !!!')
    reward *= REWARD_RATIO
    info['status'] = status
    return obs, reward, status, info

def action_rescale(action:np.ndarray, action_space, raw_action_range=(-1,1), explore:bool=True, epsilon:float=0.0):
    '''
    函数核心功能：将网络输出的动作从 [-1, 1] 映射回环境的真实动作空间。
    '''
    action = np.clip(action, *raw_action_range)
    action = action * (action_space.high - action_space.low) / 2 + (action_space.high + action_space.low) / 2
    if explore and np.random.random() < epsilon:
        action = action_space.sample()
    return action

def observation_rescale(obs):
    '''
    函数核心功能：处理图像观测维度的通道重排。
    '''
    if obs['img'] is not None:
        obs['img'] = obs['img'].transpose((2,0,1))
    return obs

class CarParkingWrapper(Wrapper):
    def __init__(self, env:CarParking, 
            action_func:callable=action_rescale, 
            reward_func:callable=reward_shaping,
            observation_func:callable=observation_rescale,
            ):
        super().__init__(env)
        self.reward_func = reward_func
        self.action_func = action_func
        self.obs_func = observation_func
        self.observation_shape = {k:self.env.observation_space[k].shape for k in self.env.observation_space}
        if 'img' in self.observation_shape:
            w,h,c = self.observation_shape['img']
            self.observation_shape['img'] = (c,w,h)

    def step(self, action=None):
        if action is None:
            return self.obs_func(self.env.step()[0])
        action = self.action_func(action, self.env.action_space)
        returns = self.env.step(action)
        obs, reward, status, info = self.reward_func(*returns)
        obs = self.obs_func(obs)
        done = False if status==Status.CONTINUE else True
        return obs, reward, done, info

    def reset(self, *args):
        obs = self.env.reset(*args)
        return self.obs_func(obs)

def reward_shaping_multi(*args, actions=None):
    '''
    函数核心功能：多车环境的联合奖励塑形。
    
    关键变量：
    - actions: 当前步的所有智能体动作列表。
    - action_reward: 针对过度转向的惩罚项。
    
    重要逻辑步骤解释：
    根据状态计算奖励基础值，并强制合并安全奖励、协同奖励以及动作平滑惩罚，确保所有的修改直接作用于强化学习计算返回值的 reward 列表中。
    '''
    from env.vehicle import VALID_STEER
    obs_list, reward_info_list, status_list, info_list = args
    reward_list = []
    
    for i in range(len(status_list)):
        status = status_list[i]
        reward_info = reward_info_list[i]
        info = info_list[i]
        
        # 初始化当前车的动作惩罚为 0
        current_action_reward = 0.0
        
        if actions is not None:
            action = actions[i]
            steer = action[0]
            # Penalize excessive steering to discourage spinning
            action_reward = - (steer / VALID_STEER[1])**2
            reward_info['action_reward'] = action_reward
            current_action_reward = action_reward
            
        if status == Status.CONTINUE:
            reward = 0
            for reward_type in REWARD_WEIGHT.keys():
                if reward_type in reward_info:
                    reward += REWARD_WEIGHT[reward_type]*reward_info[reward_type]
        elif status == Status.OUTBOUND:
            reward = -50
        elif status == Status.OUTTIME:
            reward = -1
        elif status == Status.ARRIVED:
            reward = 50
        elif status == Status.COLLIDED:
            reward = -50
        else:
            print(status)
            print('Never reach here !!!')
            
        if 'safety_reward' in reward_info:
            reward += reward_info['safety_reward']
        if 'coop_reward' in reward_info:
            reward += reward_info['coop_reward']
            
        # 【重要修复】：把计算出来的原地打转惩罚真正加到强化学习能看到的 Reward 中，而不是只存字典里
        reward += current_action_reward 
            
        reward *= REWARD_RATIO
        info['status'] = status
        reward_list.append(reward)
        
    return obs_list, reward_list, status_list, info_list

class MultiCarParkingWrapper(Wrapper):
    def __init__(self, env:MultiCarParking, 
            action_func:callable=action_rescale, 
            reward_func:callable=reward_shaping_multi,
            observation_func:callable=observation_rescale,
            ):
        super().__init__(env)
        self.reward_func = reward_func
        self.action_func = action_func
        self.obs_func = observation_func
        self.observation_shape = {k:self.env.observation_space[k].shape for k in self.env.observation_space}
        if 'img' in self.observation_shape:
            w,h,c = self.observation_shape['img']
            self.observation_shape['img'] = (c,w,h)

    def step(self, actions=None):
        if actions is None:
            return [self.obs_func(obs) for obs in self.env.step()[0]]
            
        actions_rescaled = [self.action_func(action, self.env.action_space) for action in actions]
        returns = self.env.step(actions_rescaled)
        obs_list, reward_list, status_list, info_list = self.reward_func(*returns, actions=actions_rescaled)
        obs_list = [self.obs_func(obs) for obs in obs_list]
        done_list = [False if status==Status.CONTINUE else True for status in status_list]
        return obs_list, reward_list, done_list, info_list

    def reset(self, *args):
        obs_list = self.env.reset(*args)
        return [self.obs_func(obs) for obs in obs_list]