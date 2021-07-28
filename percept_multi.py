# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:18:33 2021

@author: gtaus
"""

import numpy as np

def percept_multi(bx,fx,itr,R1,cpoint_dist,k1,k2,k3,k4):
    global human_agent_action
    
    # First Perceptual Rference
    # Distance Control
    D = abs(bx - fx)+5 # NoFrameskipe 405
    e1 = R1 - D
    R2 = e1 * k1
    # Movement Control
    MD = np.sign(bx - fx)
    e2 = R2 - MD
    if MD < 0:
        human_agent_action = 3
        R3 = e2*k2
        # Position Control
        e3 = R3 - fx
        R4 = e3*k3
        e4 = R4 + bx
        BP = e4*k4
        itr = itr + 1
        if itr > 1 or (R2 < cpoint_dist) or BP == bx:
              human_agent_action = 0
              itr = 0
    if MD > 0:
        human_agent_action = 2
        R3 = e2*k2
        # Position Control
        e3 = R3 + fx
        R4 = e3*k3
        e4 = R4 - bx
        BP = e4*k4
        itr = itr + 1
        if itr > 1 or (R2 < cpoint_dist) or BP == bx or fx > 136:
              human_agent_action = 0
              itr = 0
              
    return human_agent_action, itr