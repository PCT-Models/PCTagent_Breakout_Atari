# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:37:11 2021

@author: gtaus
"""

import numpy as np
import collections
import gym, time, cv2


# class pct(object):
    
def angle_between(center_x,center_y,touch_x,touch_y):
    delta_x = touch_x - center_x
    delta_y = center_y - touch_y
    theta_radians = np.arctan2(delta_y,delta_x)
    return np.rad2deg(theta_radians)

def slope(P11x,P11y,P22x,P22y):
    
    try:
       M =  (P22y - P11y) / (P22x - P11x)
    except ZeroDivisionError:
       M = 0
    return(M)

def y_intercept(P1x, P1y, slope):
    return P1y - slope * P1x

def line_intersect(m1, b1, m2, b2):
    if m1 == m2:
       return 5,94
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x,y


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return 0,0
    else:
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

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

def img_proc(img):
    
    yc_ball, xc_ball, hc_ball, wc_ball = 94, 9, 100, 141 
    # Crop Image
    img_ball = img[yc_ball:yc_ball+hc_ball, xc_ball:xc_ball+wc_ball]
     # Image Processing PCT - Convert it to Grayscale
    grayscale_ball = cv2.cvtColor(img_ball, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(grayscale_ball, 100, 255,cv2.THRESH_OTSU)
    bin_ball = binary
    ## Find Contours and Start Algo
    contours_ball = cv2.findContours(bin_ball, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    # x0, y0, w0, h0 = cv2.boundingRect(contours_ball[0])
    # To Reset the Game with One Contour Only
    return contours_ball
    
def ray_tracing(x_ball, y_ball, x_ball_1,y_ball_1):
    
    # A1, A2 = [x_ball, y_ball], [x_ball_1,y_ball_1]
    # B1, B2 = [-300,94], [300,94]
    # Points on Horizontal Line
    slope_A = slope(x_ball, y_ball, x_ball_1,y_ball_1)
    slope_B = slope(-300,94, 300,94)
    y_int_A = y_intercept(x_ball,y_ball, slope_A)
    y_int_B = y_intercept(-300,94, slope_B)
    B = line_intersect(slope_A, y_int_A,slope_B,y_int_B)
    # B11 = line_intersection((B1,B2), (A1,A2))
    return B
    
def bbox(thing):
    # thing = np.asarray(thing)
    bbox_plate = cv2.boundingRect(thing)
    x_plate,y_plate,w_plate,h_plate = bbox_plate
    # Paddle Coordinates
    x_cent_plate = float( x_plate  + (w_plate/2))
    y_cent_plate = float(y_plate)
    return x_cent_plate, y_cent_plate, w_plate
    
    
    
    
    
    
    
    
    
    
    
    