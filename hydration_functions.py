#!/bin/python
import os

import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PDB_small
from MDAnalysis.analysis import distances

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import os

from sympy import *
from sympy import linear_eq_to_matrix, symbols
from math import *
import math

from datetime import datetime

from multiprocessing import cpu_count
import multiprocessing
from multiprocessing import Pool
from functools import partial

def read_max_frame(f):
    mfrlist = []
    for line in f.readlines():
        fr = line.split('\t')[0]
        mfrlist.append(fr)
    maxfr0 = max(mfrlist)
    mfrlist = list(mfrlist.sort())
    for i in range(1,len(mfrlist)-1):
        if mfrlist[i-1] == mfrlist[i]-1 and mfrlist[i] == mfrlist[i+1]-1:
            maxfr = mfrlist[i]
    
    return maxfr

def eq(point1,point2,point3):
    x, y, z = symbols('x y z')
    eq = linear_eq_to_matrix([str(Plane(Point3D(point1), Point3D(point2), Point3D(point3)).equation())], [x, y, z])
    a,b,c = float(eq[0][0]),float(eq[0][1]),float(eq[0][2])
    d = float(eq[1][0]) 
    return (a,b,c,d)

#@jit(nopython=True, parallel=True)
# %%cython
def resort_points(point1,point2,point3,point4):
    a,b,c,d = eq(point1,point2,point3)
    x,y,z = point4[0],point4[1],point4[2]
    if a*x+b*y+c*z-d > 0:
        return '>'
    elif a*x+b*y+c*z-d < 0:
        return '<'
    elif a*x+b*y+c*z-d == 0:
        return '='


# %%cython    
# @jit(nopython=True, parallel=True)

def angle_btw_2vec(pointx,point1,point2):
    vec1 = point1-pointx
    vec2 = point2-pointx
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle = np.arccos(dot_product)*180/pi
    return angle

#@jit(nopython=True, parallel=True)   
# %%cython
def point_position_check(point1,point2,point3,point4,pointx): #change the order of point1,point2,point3,point4 when calling this function
    a,b,c,d = eq(point1,point2,point3)
    sign_point4 = resort_points(point1,point2,point3,point4)
    x,y,z = pointx[0],pointx[1],pointx[2]
    if sign_point4 == '>':
        if a*x+b*y+c*z-d >= 0:
            return True
        else:
            return False
            
    elif sign_point4 == '<':
        if a*x+b*y+c*z-d <= 0:
            return True
        else:
            return False

    elif sign_point4 == '=':
        pointx_triangle412 = angle_btw_2vec(pointx,point4,point2) + angle_btw_2vec(pointx,point4,point1) + angle_btw_2vec(pointx,point1,point2)
        pointx_triangle413 = angle_btw_2vec(pointx,point4,point1) + angle_btw_2vec(pointx,point4,point3) + angle_btw_2vec(pointx,point1,point3)
        pointx_triangle423 = angle_btw_2vec(pointx,point4,point2) + angle_btw_2vec(pointx,point4,point3) + angle_btw_2vec(pointx,point2,point3)
        if pointx_triangle412 == 360.0 or pointx_triangle413 == 360.0 or pointx_triangle423 == 360.0:
            return True
        else:
            return False

