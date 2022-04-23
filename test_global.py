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

from hydration_functions import *

#another method
def hydration_water_calculation2(t,u): # in one frame
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time)
        
#     u = mda.Universe('{}/npt5.gro'.format(c),
#                   '{}/5traj3_w+m-p-c_-c_prot-frskip2.xtc'.format(c))
    
    u.trajectory[t] 

    prot_heavy = u.select_atoms('(not type H) and protein',updating=True)
    water_in_5A = u.select_atoms('byres resname SOL and around 5 group prot_heavy', prot_heavy = prot_heavy, updating=True)
    hydration_water = []
    hydration_water_phi = []
        
    for w in range(0,len(water_in_5A),3):        
        ow_1 = water_in_5A[w]
        hw1_1 = water_in_5A[w+1]
        hw2_1 = water_in_5A[w+2]

        current_water = mda.AtomGroup([ow_1,hw1_1,hw2_1]) #u.select_atoms('resid ' + str(ow_1.resid) + ' and water')
        
        pow_1 = ow_1.position
        phw1_1 = hw1_1.position
        phw2_1 = hw2_1.position
        
        prot_heavy_in_5A = u.select_atoms('((not type H) and protein) and around 5 group current_water',current_water = current_water, updating=True)
        
        for atom in prot_heavy_in_5A:
            patom = atom.position
            atom = mda.AtomGroup([atom])
            water_in_5A_prot = u.select_atoms('byres resname SOL and around 5 group atom', atom = atom, updating=True)
            #waterlist = list(set(water_in_5A.resids))

            #print(ow_1,hw1_1,hw2_1)

            # waterlist_cp = waterlist.copy()
            # waterlist_cp.pop(ow_1.resid)
            
            rest_water = water_in_5A_prot - current_water
            rest_prot = prot_heavy_in_5A - atom
            rest = rest_water + rest_prot

            n = 0 # count how many atoms in water are inside the tetrehedron; if one, +1
            for ra in range(0,len(rest)):
                if n != 0:
                    break
                x = rest[ra]
                
                pointx = x.position
  
                #water plane
                position1 = point_position_check(pow_1,phw1_1,phw2_1,patom,pointx)
                #Atom-hw1-hw2
                position2 = point_position_check(phw1_1,phw2_1,patom,pow_1,pointx)
                #Atom-hw1-ow
                position3 = point_position_check(pow_1,phw1_1,patom,phw2_1,pointx)
                #Atom-hw2-ow
                position4 = point_position_check(pow_1,phw2_1,patom,phw1_1,pointx)

                if (position1 == True and position2 == True and position3 == True and position4 == True):
                    n += 1   

            if n == 0:
                if ow_1.resid not in hydration_water:
                    hydration_water.append(ow_1.resid)
                if (atom.types == 'O')[0] == True or (atom.types == 'N')[0] == True:
                    #print(atom.names)
                    # atom in protein as donor
                    H_prot = u.select_atoms('(type H and protein) and around 1.2 group atom', atom = atom,updating=True)
                    if len(H_prot) != 0:
                        for h in H_prot:
                            #print(atom.names,h.name)
                            pH_prot = h.position
                            if math.dist(patom,pow_1) <= 3.5 and angle_btw_2vec(patom,pH_prot,pow_1) <= 30:
                                if ow_1.resid not in hydration_water_phi:
                                    hydration_water_phi.append(ow_1.resid)

                    # ow in water as donor
                    if math.dist(patom,pow_1) <= 3.5 and (angle_btw_2vec(pow_1,phw1_1,patom) <= 30 or angle_btw_2vec(pow_1,phw2_1,patom) <= 30):
                        #print(atom.names,ow_1.name)
                        if ow_1.resid not in hydration_water_phi:
                            hydration_water_phi.append(ow_1.resid)

    files = ['num_hydration_water_global2_def.txt','num_hydration_water_phi_global2_def.txt']
    results = [len(hydration_water),len(hydration_water_phi)]

    for i in range(len(files)):

        with open(files[i], 'a') as file0:
            file0.writelines(str(t)+'\t'+str(results[i])+'\n')

                    
    return (len(hydration_water),hydration_water,len(hydration_water_phi),hydration_water_phi)


c0 = os.getcwd().split('/ocean/projects/mcb200065p/chy20004/210519_Hyp/whole/')[1]
u0 = mda.Universe('npt5.gro',
                  '5traj3_w+m-p-c_-c_prot-frskip2.xtc') #no need to specify c0
n_jobs = cpu_count()

if os.path.isfile('num_hydration_water_global2_def.txt') == True:
    with open('num_hydration_water_globa2l_def.txt', 'r') as file0:
        #lines = file0.readlines()
        maxfr1 = read_max_frame(file0)
    with open('num_hydration_water_phi_global2_def.txt', 'r') as file0:
        #lines = file0.readlines()
        maxfr3 = read_max_frame(file0)

    l0 = min(maxfr1,maxfr3)
    
    #old method
    # l1 = len((open('num_hydration_water_global2_def.txt', "r")).readlines())
    # #l2 = len((open('hydration_water_global2_def.txt', "r")).readlines())
    # l3 = len((open('num_hydration_water_phi_global2_def.txt', "r")).readlines())
    # #l4 = len((open('hydration_water_phi_global2_def.txt', "r")).readlines())
    # l0 = min(l1,l3)
    # if l1 > l0:
    #     with open('num_hydration_water_global2_def.txt', 'r') as file0:
    #         lines = file0.readlines()
    #         #now we have an array of lines. If we want to edit the line xyz...
    
    #     lines = lines[:l0]
    #     with open('num_hydration_water_global2_def.txt', 'w') as file0:
    #         file0.writelines(lines)
    # if l3 > l0:
    #     with open('num_hydration_water_phi_global2_def.txt', 'r') as file0:
    #         lines = file0.readlines()
    #         #now we have an array of lines. If we want to edit the line xyz...
    
    #     lines = lines[:l0]
    #     with open('num_hydration_water_phi_global2_def.txt', 'w') as file0:
    #         file0.writelines(lines)        
else:
    l0 = 0

print(l0, u0.trajectory.n_frames)
run_per_frame = partial(hydration_water_calculation2, u = u0)
frame_values = np.arange(l0, u0.trajectory.n_frames)

#if __name__ == "__main__":
with Pool(n_jobs) as worker_pool:
    result = worker_pool.map(run_per_frame, frame_values)

files = ['num_hydration_water_global2.txt','num_hydration_water_phi_global2.txt']
for j in range(len(result)):
    for i in range(len(files)):
        with open(files[i], 'a') as file0:
            if i == 0:
                file0.writelines(str(result[j][0])+'\n')
            else:
                file0.writelines(str(result[j][2])+'\n')

