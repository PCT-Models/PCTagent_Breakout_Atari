# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:43:05 2021

@author: gtaus
"""

def percept_double(bx,itr,dist_vib,x_cent_plate):
    global human_agent_action
    
    if (bx - x_cent_plate) < 0:
        human_agent_action = 3
        itr = itr + 1
        if itr > 1 or abs(bx - x_cent_plate) < dist_vib:
              human_agent_action = 0
              itr = 0
    if (bx - x_cent_plate) > 0:
        human_agent_action = 2
        itr = itr + 1
        if itr > 1 or abs(bx - x_cent_plate) < dist_vib:
               human_agent_action = 0
               itr = 0
    
    return human_agent_action, itr