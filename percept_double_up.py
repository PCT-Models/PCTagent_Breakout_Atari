# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 20:21:02 2021

@author: gtaus
"""
def percept_double_up(bx,fx,itr,dist_vib):
    global human_agent_action
    
    if (bx - fx) < 0:
        human_agent_action = 3
        itr = itr + 1
        if itr > 1 or abs(bx - fx) < dist_vib:
              human_agent_action = 0
              itr = 0
    if (bx - fx) > 0:
        human_agent_action = 2
        itr = itr + 1
        if itr > 1 or abs(bx - fx) < dist_vib or fx > 136:
               human_agent_action = 0
               itr = 0
    
    return human_agent_action, itr