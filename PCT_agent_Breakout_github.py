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

env = gym.make('BreakoutNoFrameskip-v4' if len(sys.argv)<2 else sys.argv[1])

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



x_balli = collections.deque([0] * 2, maxlen=2)
y_balli = collections.deque([0] * 2, maxlen=2)
x_platei = collections.deque([0] * 2, maxlen=2)
y_platei = collections.deque([0] * 2, maxlen=2)
rewardsi = collections.deque([0] * 100, maxlen=100)

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
            # print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
            
            
        else:
            skip -= 1

        

        obser, r, done, info = env.step(a)
        # if r != 0:
       
        total_reward += r
        
        # Put latest 20 rewards in a list
        rewardsi.append(total_reward)
        
        
        
        # print("reward %d" % total_reward)
        window_still_open = env.render()
        img = env.render(mode="rgb_array")
        
        
        # For Ball to detect crop Image Just under the Bricks
        yc_ball, xc_ball, hc_ball, wc_ball = 94, 9, 100, 141 
        
        # Crop Image
        img_ball = img[yc_ball:yc_ball+hc_ball, xc_ball:xc_ball+wc_ball]
        
        
        
         # Image Processing PCT - Convert it to Grayscale
        grayscale_ball = cv2.cvtColor(img_ball, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(grayscale_ball, 100, 255,cv2.THRESH_OTSU)
        bin_ball = binary
        
        ## Find Contours and Start Algo
        contours_ball = cv2.findContours(bin_ball, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        x0, y0, w0, h0 = cv2.boundingRect(contours_ball[0])
        cv2.rectangle(img_ball,(x0,y0), (x0+w0,y0+h0), (0,255,0), 1)
        a, b = round(x0+w0/2), round(y0)
        cv2.circle(img_ball, (a,b), radius=1, color=(0, 0, 255), thickness=-1)
        
        if len(contours_ball) == 2:
            x1, y1, w1, h1 = cv2.boundingRect(contours_ball[1])
            a1, b1 = round(x1+w1/2), round(y1+h1)
            cv2.circle(img_ball, (a1,b1), radius=1, color=(0, 0, 255), thickness=-1)
            cv2.rectangle(img_ball,(x1,y1), (x1+w1,y1+h1), (0,255,0), 1)
        
               
        
        # To Reset the Game with One Contour Only
        if len(contours_ball) == 0:
            print('Nothing is Detecte Something Wrong')
            continue
        elif len(contours_ball) == 1:
            cnt_plate = contours_ball[0]
            ## Get the Plate's bounding rect
            bbox_plate = cv2.boundingRect(cnt_plate)
            # print(bbox_plate)
            x_plate,y_plate,w_plate,h_plate = bbox_plate
            # Plate Coordinates
            x_cent_plate = float( x_plate  + (w_plate/2))
            y_plate = float(y_plate)
            x_platei.append(x_cent_plate)
            y_platei.append(y_plate)
            
            human_agent_action = 1
            
            
            
        elif len(contours_ball) == 2:
            cnt_ball = contours_ball[1]
            ## Get the Ball's bounding rect
            bbox_ball = cv2.boundingRect(cnt_ball)
            x_ball,y_ball,w_ball,h_ball = bbox_ball
              # Ball Coordinates
            x_ball = float(x_ball+(w_ball/2))
            y_ball = float(y_ball)
            x_balli.append(x_ball)
            y_balli.append(y_ball)
            # print(y_ball)
            
            cnt_plate = contours_ball[0]
            ## Get the Plate's bounding rect
            bbox_plate = cv2.boundingRect(cnt_plate)
            # print(bbox_plate)
            x_plate,y_plate,w_plate,h_plate = bbox_plate
            # Plate Coordinates
            x_cent_plate = float(x_plate + (w_plate/2))
            y_plate = float(y_plate)
            x_platei.append(x_cent_plate)
            y_platei.append(y_plate)
            
            
            
            
            
            # Perceptual Ray Tracing
            # Parameters
            A1, A2 = [x_ball, y_ball], [x_balli[0],y_balli[0]]
            B1, B2 = [-300,94], [300,94]
            R1, R2 = [140,94], [140,0]
            L1, L2 = [0,94], [0,-94]
            
            cv2.line(img_ball,B1,B2,(0,0,255), 1)
            cv2.line(img_ball,R1,R2,(0,0,255), 1)
            cv2.line(img_ball,L1,L2,(0,0,255), 1)
            
            
            base = x_ball
            
            # Distance Between Ball and the Plate
            cpoint_dist = (w_plate/2)-0.25
            # Center Point for Ball to Maintain
            cpoint = 70
            dist_vib = (w_plate/2)
            
            
            
            if total_reward > -1:
                
                # If Ball going Down 
                if ( ((94-y_ball) - (94-y_balli[0]) < 0) ):
                
                            # Hiearchical Loop
                                                        
                            # First Perceptual Estimation
                            R1 = 0
                            # All Gains
                            k1 = -1
                            k2 = 1
                            k3 = 1
                            k4 = 1
                            # Distance Control
                            D = abs(base - x_cent_plate)+5
                            e1 = R1 - D
                            R2 = e1 * k1
                            
                            # Movement Control
                            MD = np.sign(base - x_cent_plate)
                            e2 = R2 - MD
                            
                            if MD < 0:
                                human_agent_action = 3
                                R3 = e2*k2
                                # Position Control
                                e3 = R3 - x_cent_plate
                                R4 = e3*k3
                                e4 = R4 + base
                                BP = e4*k4
                                
                                itr = itr + 1
                                if itr > 1 or (R2 < cpoint_dist) or BP == base:
                                      human_agent_action = 0
                                      itr = 0
                                      
                            if MD > 0:
                                human_agent_action = 2
                                R3 = e2*k2
                                # Position Control
                                e3 = R3 + x_cent_plate
                                R4 = e3*k3
                                e4 = R4 - base
                                BP = e4*k4
                                
                                itr = itr + 1
                                if itr > 1 or (R2 < cpoint_dist) or BP == base or x_cent_plate > 136:
                                      human_agent_action = 0
                                      itr = 0
                                      
                            
                # Strategy to come in the Middle
                # If Ball Going up Come in the Center
                if ((94-y_ball) - (94-y_balli[0])) > 0:

                            if (base - x_cent_plate) < 0:
                                human_agent_action = 3
                                itr = itr + 1
                                if itr > 1 or abs(cpoint - x_cent_plate) < dist_vib:
                                # if itr > 1 or abs(base - x_cent_plate) < dist_vib:
                                      human_agent_action = 0
                                      itr = 0
    
                            if (base - x_cent_plate) > 0:
                                    human_agent_action = 2
                                    itr = itr + 1
                                    if itr > 1 or abs(cpoint - x_cent_plate) < dist_vib:
                                    # if itr > 1 or abs(base - x_cent_plate) < dist_vib:
                                          human_agent_action = 0
                                          itr = 0
                                          
            # Green Dot on the Line
            cv2.circle(img_ball, (round(x_ball),94), radius=1, color=(0, 255, 0), thickness=-1)
            imgS = cv2.resize(img, (400,700))
            cv2.imshow('pctAgent', imgS)

        #------------OLD CODE-------------#
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        
        time.sleep(0.001)
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
