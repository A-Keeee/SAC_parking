import sys
sys.path.append("/home/ake/1-500/HOPE_ge/src")
from math import pi, cos, sin
import numpy as np
from numpy.random import randn, random
from shapely.geometry import LinearRing
from env.vehicle import State
from env.map_base import Area
from configs import *

def random_gaussian_num(mean, std, clip_low, clip_high):
    rand_num = randn()*std + mean
    return np.clip(rand_num, clip_low, clip_high)

def random_uniform_num(clip_low, clip_high):
    rand_num = random()*(clip_high - clip_low) + clip_low
    return rand_num

def generate_multi_car_bay_parking(map_level, num_agents=NUM_AGENTS):
    origin = (0., 0.)
    bay_half_len = 30.0 # Make it wide enough for 3 cars and spaces
    bay_PARK_WALL_DIST = BAY_PARK_WALL_DIST_DICT[map_level]
    
    obstacle_back = LinearRing(( 
        (origin[0]+bay_half_len, origin[1]),
        (origin[0]+bay_half_len, origin[1]-1), 
        (origin[0]-bay_half_len, origin[1]-1), 
        (origin[0]-bay_half_len, origin[1])))
        
    generate_success = True
    dests = []
    dest_boxes = []
    
    # We will put cars in a row with spacing
    car_spacing = WIDTH + 1.2 # roughly a parking space width
    
    # Randomly pick 3 indices out of e.g., 10 possible spots, ensure they are not adjacent to avoid being too cramped, 
    # but actually the prompt says "三个随机的目标车位", so they can be anywhere. Let's just pick 3 unique indices from 0 to 9.
    import random as pyrand
    spots = pyrand.sample(range(8), num_agents)
    spots.sort()
    
    obstacles = [obstacle_back]
    
    # Generate destinations
    for i in range(num_agents):
        dest_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
        dest_x = origin[0] - 10.0 + spots[i] * (car_spacing + 0.5)
        # Calculate Y
        rb, _, _, lb  = list(State([dest_x, origin[1], dest_yaw, 0, 0]).create_box().coords)[:-1]
        min_dest_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
        dest_y = random_gaussian_num(min_dest_y+0.4, 0.2, min_dest_y, min_dest_y+0.8)
        
        dest_state = [dest_x, dest_y, dest_yaw]
        dests.append(dest_state)
        car_rb, car_rf, car_lf, car_lb  = list(State([dest_x, dest_y, dest_yaw, 0, 0]).create_box().coords)[:-1]
        dest_boxes.append(LinearRing((car_rb, car_rf, car_lf, car_lb)))
        
    # Generate obstacles (static parked cars) in the remaining spots
    for i in range(8):
        if i not in spots:
            if pyrand.random() < 0.6: # 60% chance to have a parked car
                obs_x = origin[0] - 10.0 + i * (car_spacing + 0.5)
                obs_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
                rb, _, _, lb  = list(State([obs_x, origin[1], obs_yaw, 0, 0]).create_box().coords)[:-1]
                min_obs_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
                obs_y = random_gaussian_num(min_obs_y+0.4, 0.2, min_obs_y, min_obs_y+0.8)
                obstacles.append(State([obs_x, obs_y, obs_yaw, 0, 0]).create_box())
                
    # Walls on left and right
    left_wall = LinearRing((
        (origin[0]-bay_half_len, origin[1]),
        (origin[0]-bay_half_len, origin[1]+bay_PARK_WALL_DIST),
        (origin[0]-bay_half_len-1, origin[1]+bay_PARK_WALL_DIST),
        (origin[0]-bay_half_len-1, origin[1])
    ))
    right_wall = LinearRing((
        (origin[0]+bay_half_len, origin[1]),
        (origin[0]+bay_half_len, origin[1]+bay_PARK_WALL_DIST),
        (origin[0]+bay_half_len+1, origin[1]+bay_PARK_WALL_DIST),
        (origin[0]+bay_half_len+1, origin[1])
    ))
    obstacles.append(left_wall)
    obstacles.append(right_wall)

    # Generate starts
    starts = []
    start_boxes = []
    max_obstacle_y = max([np.max(np.array(obs.coords)[:,1]) for obs in obstacles]) + MIN_DIST_TO_OBST
    
    valid_start_x_range = (origin[0]-bay_half_len/2, origin[0]+bay_half_len/2)
    valid_start_y_range = (max_obstacle_y+1, bay_PARK_WALL_DIST+max_obstacle_y-1)
    
    for i in range(num_agents):
        start_box_valid = False
        attempts = 0
        while not start_box_valid and attempts < 100:
            start_box_valid = True
            attempts += 1
            start_x = random_uniform_num(*valid_start_x_range)
            start_y = random_uniform_num(*valid_start_y_range)
            start_yaw = random_gaussian_num(0, pi/6, -pi/2, pi/2)
            start_yaw = start_yaw+pi if random()<0.5 else start_yaw
            start_box = State([start_x, start_y, start_yaw, 0, 0]).create_box()
            
            # Check collision with obstacles
            for obst in obstacles:
                if obst.intersects(start_box):
                    start_box_valid = False
                    break
            
            # Check overlap with ALL dest boxes
            if start_box_valid:
                for d_box in dest_boxes:
                    if d_box.intersects(start_box):
                        start_box_valid = False
                        break
                        
            # Check overlap with other start boxes
            if start_box_valid:
                for s_box in start_boxes:
                    if s_box.intersects(start_box):
                        start_box_valid = False
                        break
                        
            if start_box_valid:
                starts.append([start_x, start_y, start_yaw])
                start_boxes.append(start_box)
        if not start_box_valid:
            generate_success = False

    if generate_success:
        return starts, dests, obstacles
    else:
        return generate_multi_car_bay_parking(map_level, num_agents)

starts, dests, obs = generate_multi_car_bay_parking("Normal", 3)
print("Success!")
print(len(starts), len(dests), len(obs))
