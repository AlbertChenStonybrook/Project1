# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 23:06:26 2021

@author: Albert
"""
import numpy as np
from args import *
from src.fcn import *
from src.trainer import *
from src.utils import *
from src.selector import *
from Pre_train import pretrain
from agents.pdqn import PDQNAgent
from agents.pdqn_split import SplitPDQNAgent
from agents.pdqn_multipass import MultiPassPDQNAgent
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from agents.pdqn import PDQNAgent

#from Env import Env

from Env_val import Env_val
from tqdm import tqdm


def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32) for i in range(44)]
    params[act][:] = act_param
    return (act, params)




def validate(agent,observation_dim,action_dim,balance_factor,max_steps,model,X_val,Y_val,val_size):
    feature_selector = FeatureSelector(X_val, args.gen_gamma) if args.gen_gamma > 0.0 else None
    env_val=Env_val(observation_dim,action_dim,model,balance_factor,max_steps,feature_selector)

    acc=[]
    for k in range(X_val.shape[0]):
        x=X_val[k,:]
        y=Y_val[k]
        #print("x",x)
        x = torch.FloatTensor(x)
        x_var = Variable(x,requires_grad=True).type(torch.FloatTensor)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            x_var = x_var.cuda()
        f=model.forward(x_var).data.cpu().numpy().flatten()
        env_val.reset(x,y)
        
        
        #env.reset(x,y)
        state = np.array(x, dtype=np.float32, copy=False)
        act_ava=env_val.get_act_ava()
        
        
        act, act_param, all_action_parameters = agent.act(state,act_ava)
        #print("act",act,"act_param",act_param)
        action = pad_action(act, act_param)
        for j in range(max_steps):
            #print("act",act)
            #print("act",act)
            ret = env_val.step(action)
            act_ava=env_val.get_act_ava()
            #print("act_ava",act_ava)
            (next_state, steps), terminal = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            next_act, next_act_param, next_all_action_parameters = agent.act(next_state,act_ava)
            #print("next_act",next_act,"param",next_act_param)
            next_action = pad_action(next_act, next_act_param)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            #print("act",act,"act_param",act_param)
            
            action = next_action
            state = next_state
            if terminal:
                break
        if(terminal):
            acc.append(1)
        else:
            acc.append(0)
       
    return(np.sum(acc)/len(acc))
        
