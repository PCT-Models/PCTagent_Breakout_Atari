# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 01:12:23 2021

@author: Tauseef Gulrez and Warren Mansell
Melbourne, Australia and Manchester, UK

"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 23:11:13 2021

@author: gtaus
"""

#!/usr/bin/env python
import sys
sys.path.append('D:/Anaconda/envs/spyder/Lib/site-packages')
import gym, time, cv2
import numpy as np
import collections


# Image Processing Needs to be improved
from pct_class import img_proc, ray_tracing, bbox
from percept_multi import percept_multi
from percept_double import percept_double
from percept_double_up import percept_double_up

# env = gym.make('Breakout-v0' if len(sys.argv)<2 else sys.argv[1])
# env = gym.make('Breakout-v1' if len(sys.argv)<2 else sys.argv[1])
# env = gym.make('Breakout-v4' if len(sys.argv)<2 else sys.argv[1])
# env = gym.make('BreakoutDeterministic-v0' if len(sys.argv)<2 else sys.argv[1])
# env = gym.make('BreakoutDeterministic-v4' if len(sys.argv)<2 else sys.argv[1])
# env = gym.make('BreakoutNoFrameskip-v0' if len(sys.argv)<2 else sys.argv[1])
env = gym.make('BreakoutNoFrameskip-v4' if len(sys.argv)<2 else sys.argv[1])

# To save Rewards
# outF = open("results_Breakout-V4_500.txt", "a")

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0
human_agent_action = 0
human_wants_restart = False
human_sets_pause = False


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: 
        human_wants_restart = True
    if key==32: 
        human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

# Saving Paddle and Ball Positions
x_balli = collections.deque([0] * 2, maxlen=2)
y_balli = collections.deque([0] * 2, maxlen=2)
x_platei = collections.deque([0] * 2, maxlen=2)
y_platei = collections.deque([0] * 2, maxlen=2)

no_of_games = 0

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    itr = 0
    
    # Game's Main Loop
    while 1:
        if not skip:
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1
        obser, r, done, info = env.step(a)
        # if r != 0:
        total_reward += r
        window_still_open = env.render()
        img = env.render(mode="rgb_array")

        # Image Processing of the Environment
        contours_ball = img_proc(img)

        # Generate Points of the Detected Objects
        if len(contours_ball) == 0:
            print('Nothing is Detecte Something Wrong')
            continue
        elif len(contours_ball) == 1:
            x_cent_plate, y_cent_plate, w_plate = bbox(contours_ball[0])
            x_platei.append(x_cent_plate)
            y_platei.append(y_cent_plate)
            human_agent_action = 1

        elif len(contours_ball) == 2:
            x_ball, y_ball, w_ball = bbox(contours_ball[1])
            x_cent_plate, y_cent_plate, w_plate = bbox(contours_ball[0])
            # Store everything as a two array matrix
            x_balli.append(x_ball)
            y_balli.append(y_ball)
            x_platei.append(x_cent_plate)
            y_platei.append(y_cent_plate)
            # # Perceptual Ray Tracing
            B = ray_tracing(x_ball, y_ball, x_balli[0],y_balli[0])
            # Distance Between Ball and the Plate
            cpoint_dist = (w_plate/2) - 0.25
            # Ball and Paddle Previous Positions
            bx = B[0]
            fx = x_cent_plate
            # First Perceptual Input
            R1 = 0
            # All Gains
            k1, k2, k3, k4 = -1, 1, 1, 1
            dist_vib = (w_plate/2)
            dist_vib1 = 6
            
            if total_reward > -1:
                # If Ball going Down 
                if ( ((94-y_ball) - (94-y_balli[0]) < 0) ):
                    # Multi-Hierarchical PCT Mode
                    # human_agent_action, itr = percept_multi(bx,fx,itr,R1,cpoint_dist,k1,k2,k3,k4)
                    # Double-Hierarchical Mode (Requires T=Ray Tracing)
                    human_agent_action, itr = percept_double_up(bx,fx,itr,dist_vib1)

                # Image Processing Needs to be done
                # If Ball Going up Track it
                bx1 = 70
                if (((94-y_ball) - (94-y_balli[0])) > 0):
                    human_agent_action, itr = percept_double(bx1,itr,dist_vib,x_cent_plate)

        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        
        time.sleep(0.01)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    no_of_games = no_of_games + 1
    window_still_open = rollout(env)
    print(no_of_games)
    if window_still_open==False: 
        # outF.close()
        cv2.destroyAllWindows()
        break
