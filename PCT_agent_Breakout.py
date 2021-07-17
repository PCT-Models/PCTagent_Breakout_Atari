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
from numpy import random


# env = gym.make('Breakout-v0' if len(sys.argv)<2 else sys.argv[1])
# env = gym.make('Breakout-v1' if len(sys.argv)<2 else sys.argv[1])
# env = gym.make('Breakout-v4' if len(sys.argv)<2 else sys.argv[1])
# env = gym.make('BreakoutDeterministic-v1' if len(sys.argv)<2 else sys.argv[1])
# env = gym.make('BreakoutDeterministic-v4' if len(sys.argv)<2 else sys.argv[1])
env = gym.make('BreakoutNoFrameskip-v0' if len(sys.argv)<2 else sys.argv[1])


# To save Rewards
# outF = open("results_Breakout-V4_500.txt", "a")


if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    
human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def noise():
    x,x1,x2,x3,x4,nu = 0,0,0,0,0,0
    d = collections.deque([0] * 800, maxlen=800)
    constant = 5.0; maxx = 0; 
    for i in range(0,399):
        x =  0.5 - np.random.rand()
        x1 = x1+(x-x1)/constant
        x2 = x2+(x1-x2)/constant
        x3 = x3+(x2-x3)/constant
        x4 = x4+(x3-x4)/constant
    for i in range(0,799):
        x = 0.5 - np.random.rand()
        x1 = x1+(x-x1)/constant
        x2 = x2+(x1-x2)/constant
        x3 = x3+(x2-x3)/constant
        x4 = x4+(x3-x4)/constant
        d.append(10000*x4)
    for i in range(0,799):
        if (d[i] > maxx):
            maxx = np.abs(d[i])
    for i in range(0,799):
        d[i]  = d[i]/(maxx) * 100.0
    return d

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
            #cv2.imshow('PCT Window', grayscale_ball)
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
            
            # Ball and Drawing Parameters
            A1, A2 = [x_ball, y_ball], [x_balli[0],y_balli[0]]
            B1, B2 = [-300,94], [300,94]
            R1, R2 = [140,94], [140,0]
            L1, L2 = [0,94], [0,-94]
            cv2.line(img_ball,B1,B2,(0,0,255), 1)
            cv2.line(img_ball,R1,R2,(0,0,255), 1)
            cv2.line(img_ball,L1,L2,(0,0,255), 1)
            cpoint_dist = (w_plate/2)-0.25
            # PCT Initial Variables
            i,l,bhp,bvp,fx,fz,fzp,fy,up,vp,ldp,fvel,facc,lvel,lacc = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 
            dt = 0.05
            # Ball Velocity in X-axis
            cv = np.abs((x_ball - x_balli[1]) / 0.001)
            angley = (np.pi/2.0)/(np.random.rand()*0.8+1.2);
            anglez = ((np.random.rand()*200)/100.0)*(np.pi/8.0)-np.pi/8.0;
            hc = (np.cos (angley)* cv); # Horizonal movement
            vc = (np.sin (angley)* cv); # Vertical movement
            # fx = ((Math.random()*100-50.0)/50.0)*6.0+45.0;
            fx = x_cent_plate
            # PCT Gains
            fgain, lgain, fslow, lslow = 2000., 2000, 500, 1000
            ref1= 0.0185 
            ref2 = 0
            d, g = 1, 10.0 
            # Delays
            delay = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]];
            drag = 0
            if total_reward > -1:
                # Hiearchical Loop
                bh = bhp + hc*dt # dt,hc are constants
                bhp = bh
                bv = bvp + vc*dt # vc,dt are constants
                bvp = bv
                #Iterating over velocities going backwards
                vc = vc - (g*dt)     # sin of Velocity
                hc = hc - (drag*dt)  # cos of Velocity
                # bx = np.cos(anglez)*bh
                # bz = np.sin(anglez)*bh
                bx = x_ball
                # bx = B[0]
                bz = y_ball
                by = bv
                # Avoid Infinity
                try:
                    ld =  (fz - bz) / (fx - bx)
                except ZeroDivisionError:
                    ld = 0
                #v = vertical angle
                xprime = np.sqrt((fx-bx)*(fx-bx)+(fz-bz)*(fz-bz));
                yprime = by;
                v = yprime/xprime;
                p1 = v-vp 
                p2 = ld-ldp
                vp = v
                ldp = ld
                for j in range(0,2):
                  delay [1][j+1] = delay [1][j];
                  delay [2][j+1] = delay [2][j];
                noisey = noise()
                delay [1][0] = p1 + noisey[1]/30000.0
                delay [2][0] = p2 + noisey[1]/30000.0
                l = l+ 5 
                if (l>399): 
                    l=0
                fvel = fvel + (fgain * (ref1-delay[1][1])-fvel)/fslow
                lvel = lvel + (lgain * (ref2- delay[2][1])-lvel)/lslow
                # if (lvel>6.0):
                #    lvel = 6.0
                # if (lvel<-6.0):
                #    lvel = -6.0
                # if (fvel>6.0):
                #    fvel = 6.0
                # if (fvel<-6.0):
                #    fvel = -6.0
                fx = fx - fvel 
                fz = fz + lvel
                # If Ball going Down 
                if ( ((94-y_ball) - (94-y_balli[0]) < 0) ):
                            if (bx - fx) < 0:
                                  human_agent_action = 3
                                  if (abs((bx - fx))) < cpoint_dist:
                                      human_agent_action = 0
                            # Hierarchical Loop
                            # # Ball going Right Side - Plate is on Left and Distance is small
                            if (bx - fx) > 0 :
                                  human_agent_action = 2
                                  if (abs((bx - fx))) < cpoint_dist:
                                      human_agent_action = 0
                # Image Processing Needs to be done
                # If Ball Going up Track it
                if ((94-y_ball) - (94-y_balli[0])) > 0:
                            
                            # bx = x_ball[0]
                            if (bx - fx) < 0:
                                human_agent_action = 3
                                if  abs(bx - fx) < cpoint_dist:
                                    human_agent_action = 0
                            if (bx - fx) > 0:
                                human_agent_action = 2
                                if abs(bx - fx) < cpoint_dist:
                                    human_agent_action = 0
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
    # outF.write(str(total_reward)+"\n")
    
    
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
