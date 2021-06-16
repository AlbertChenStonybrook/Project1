# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:53:44 2021

@author: Albert
"""
from copy import deepcopy
from torch.autograd import Variable
import torch as torch
import numpy as np
from args import *
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class Env():
    def __init__(self,observation_dim,action_dim,max_step,X_tr,y_tr):
        self.act_dim=action_dim
        self.obs_dim=observation_dim
        self.timestep=0
        self.max_step=max_step
        device="cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.X_tr=X_tr
        self.Y_tr=y_tr
    def reset(self,v1,v2):
        self.v1_ori=deepcopy(v1)
        self.v2_ori=deepcopy(v2)
        self.state=deepcopy(np.array([v1,v2]))
        self.action_available=np.arange(self.act_dim)
        #print("action_ava",self.action_available)
        return self.state

    def step(self,action):
        self.timestep+=1
        i=action[0]
        x=action[1][i][0]
        if(i in self.action_available):
            #print("TRUE")
            t=np.arange(len(self.action_available))[np.where(self.action_available==i)][0]
            self.action_available=np.delete(self.action_available,t)
            #print("diff",diff,"new",self.state[i],"ori",self.x_ori[i])
            self.state[i]=self.state[i]+x
            if(self.timestep==2):
                terminal=True
                reward=self.reward()
            else:
                terminal=False
                reward=0
            #print("REWARD",reward)
            return ((self.state,self.timestep),reward,terminal)
            
    def reward(self):
        for i in range(self.X_tr.shape[0]):
            if
    def get_act_ava(self):
        act_ava=deepcopy(self.action_available)
        return act_ava
    
    
        
    