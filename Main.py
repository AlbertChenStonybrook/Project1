# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:56:09 2021

@author: Albert
"""
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
from copy import deepcopy

#from Env import Env
import numpy as np
from Env import Env
from Env_val_once import Env_val
from tqdm import tqdm
from Validate import validate



def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32) for i in range(action_dim)]
    params[act][:] = act_param
    return (act, params)

def load_model(train_data, args):
    num_feat = train_data.getX().shape[1]
    num_class = len(np.unique(train_data.gety()))
    scaler = StandardScaler(with_std=True)
    scaler.fit(train_data.getX())
    stds = np.sqrt(scaler.var_)
    if args.model_scaler:
        model = FCN(num_feat, num_class, args.hiddens, scaler.mean_, stds)
    else:
        model = FCN(num_feat, num_class, args.hiddens)
    return model

def Evaluation(X_test,model):
    use_cuda = torch.cuda.is_available()
    for i in range(X_test.shape[0]):
        x=X_tr[i,:]
        x_var=Variable(deepcopy(x),requires_grad=False).type(torch.FloatTensor)
        if use_cuda:
            x_var = x_var.cuda()
        f=model.forward(x_var).data.cpu().numpy().flatten()
        I = f.argsort()[::-1]
        y=I[0]
        for j in range(max_steps):
            #print("act",act)
            ret = env.step(action)
            act_ava=env.get_act_ava()
            #print("act_ava",act_ava)
            (next_state, steps), reward, terminal = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            next_act, next_act_param, next_all_action_parameters = agent.act(next_state,act_ava)
            next_action = pad_action(next_act, next_act_param)
            agent.step(state, (act, all_action_parameters), reward, next_state,
                       (next_act, next_all_action_parameters), terminal, steps)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            #print("action",len(action)
            #print()
            state = next_state
            episode_reward+=reward
            if terminal:
                break
        
        
        

#def 

seed=np.random.randint(100000)
evaluation_episodes=1000
episodes=50000
batch_size=512
gamma=0.99
inverting_gradients=True,            
initial_memory_threshold=18250
use_ornstein_noise=False             
replay_memory_size=50000
epsilon_steps=30000
epsilon_final=0.56
tau_actor=0.1
tau_actor_param=0.0005
learning_rate_actor=1e-3
learning_rate_actor_param=1e-5
scale_actions=True
initialise_params=True
clip_grad=15.
split=False
multipass=True
indexed=False
weighted=False
average=False
random_weighted=False
zero_index_gradients=False
action_input_layer=0
layers=[128]
save_freq=0
save_dir="results/platform"
render_freq=100
save_frames=False
visualise=True
title="PDDQN"
balance_factor=0.18
n_neighbour=20




###Model Pretrain
pretrain()


###Data Processing
scaler, le, _, _, features, train_data, val_data, test_data = read_data(
        args.csv, args.seed, scaler=args.pre_scaler)
model = load_model(train_data, args)
model.load_state_dict(torch.load(args.model_temp_path))
trainer = Trainer(model)
num_action = train_data.getX().shape[1]
bound_min, bound_max, bound_type = get_constraints(train_data.getX())

###RL setting
from agents.pdqn import PDQNAgent
observation_dim=train_data.getX().shape[1]
action_dim=train_data.getX().shape[1]
action_high=bound_max
action_low=bound_min
X_tr=train_data.getX()
Y_tr=train_data.gety()
tr_size=X_tr.shape[0]
X_val=val_data.getX()
Y_val=val_data.gety()
val_size=X_val.shape[0]
X_test=test_data.getX()
Y_test=test_data.gety()
test_size=X_test.shape[0]


if save_freq > 0 and save_dir:
   save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
   os.makedirs(save_dir, exist_ok=True)
   
   
   
###Initialize weight Lack
        
####RL framework
np.random.seed(seed)
max_steps = 5
total_reward = 0.
returns = []
label=[]
acc_tr=[]
reward_record_tr=[]
acc_val=[]
acc_test=[]
acc_val_best=float("-inf")
acc_test_best=float("-inf")
#reward_record_val=[]


##agent
agent_class = PDQNAgent

agent = agent_class(
                   observation_dim, action_dim,action_high,action_low,
                   batch_size=batch_size,
                   learning_rate_actor=learning_rate_actor,
                   learning_rate_actor_param=learning_rate_actor_param,
                   epsilon_steps=epsilon_steps,
                   gamma=gamma,
                   tau_actor=tau_actor,
                   tau_actor_param=tau_actor_param,
                   clip_grad=clip_grad,
                   indexed=indexed,
                   weighted=weighted,
                   average=average,
                   random_weighted=random_weighted,
                   initial_memory_threshold=initial_memory_threshold,
                   use_ornstein_noise=use_ornstein_noise,
                   replay_memory_size=replay_memory_size,
                   epsilon_final=epsilon_final,
                   inverting_gradients=inverting_gradients,
                   actor_kwargs={'hidden_layers': layers,
                                 'action_input_layer': action_input_layer,},
                   actor_param_kwargs={'hidden_layers': layers,
                                       'squashing_function': False,
                                       'output_layer_init_std': 0.0001,},
                   zero_index_gradients=zero_index_gradients,
                   seed=seed)
agent_val=deepcopy(agent)
agent_val.epsilon=0

feature_selector = FeatureSelector(X_tr, args.gen_gamma) if args.gen_gamma > 0.0 else None
env=Env(observation_dim,action_dim,model,balance_factor,max_steps,feature_selector)
env.build_feature_local(X_tr,Y_tr,n_neighbour)

count=0
for k in range(episodes):
    episode_reward=0
    i=np.random.randint(tr_size)
    #i=6
    x=X_tr[i,:]
    y=Y_tr[i]
    
    x = torch.FloatTensor(x)
    x_var = Variable(x,requires_grad=True).type(torch.FloatTensor)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        x_var = x_var.cuda()
    f=model.forward(x_var).data.cpu().numpy().flatten()
    env.reset_local(x,y,i)
    
    
    #env.reset(x,y)
    state = np.array(x, dtype=np.float32, copy=False)
    act_ava=env.get_act_ava()
    
    
    act, act_param, all_action_parameters = agent.act(state,act_ava)
    act_param=min(act_param,args.action_bound)
    action = pad_action(act, act_param)
    for j in range(max_steps):
        #print("act",act)
        ret = env.step(action)
        act_ava=env.get_act_ava()
        #print("act_ava",act_ava)
        (next_state, steps), reward, terminal = ret
        next_state = np.array(next_state, dtype=np.float32, copy=False)
        next_act, next_act_param, next_all_action_parameters = agent.act(next_state,act_ava)
        next_act_param=min(next_act_param,args.action_bound)
        next_action = pad_action(next_act, next_act_param)
        agent.step(state, (act, all_action_parameters), reward, next_state,
                   (next_act, next_all_action_parameters), terminal, steps)
        act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
        action = next_action
        #print("action",len(action)
        #print()
        state = next_state
        episode_reward+=reward
        if terminal:
            break
    #print("terminal",terminal)
    if(terminal):
        label.append(1)
    else:
        label.append(0)
    agent.end_episode()
    returns.append(episode_reward)
    total_reward+=episode_reward
    if k % 100 == 0:
            #agent.save_models("temp")
            #agent_val.load_models("temp")
            #print("finish",k)
            agent_val.copy_models(agent.actor,agent.actor_param)
            acc_tr.append(np.array(label[-100:]).mean())
            reward_record_tr.append(np.array(returns[-100:]).mean())
            np.savetxt("acc_tr.txt",acc_tr)
            np.savetxt("reward_tr.txt",reward_record_tr)
            if(k%300==0 and k>2000):
                r_val=validate(agent_val,observation_dim,action_dim,balance_factor,max_steps,model,X_val,Y_val,val_size)
                acc_val.append(r_val)
                np.savetxt("acc_val.txt",acc_val)
                r_test=validate(agent_val,observation_dim,action_dim,balance_factor,max_steps,model,X_test,Y_test,val_size)
                if(r_test>acc_test_best):
                    agent_val.save_models("best")
                    acc_test_best=r_test
                #agent_val.save_models("model"+str(k))
                agent_val.save_models("model"+str(k))
                acc_test.append(r_test)
                np.savetxt("acc_test.txt",acc_test)
                print('{0:5s} R:{1:.4f} r100:{2:.4f} rtr{3:.2f} rval{4:.2f} rtest{5:.2f}'.format(str(k), total_reward / (k + 1), np.array(returns[-100:]).mean(),np.array(label[-100:]).mean(),r_val,r_test))
            elif(k%100==0):
                print('{0:5s} R:{1:.4f} r100:{2:.4f} rtr{3:.2f}'.format(str(k), total_reward / (k + 1), np.array(returns[-100:]).mean(),np.array(label[-100:]).mean()))
            else:
                pass
 





#print('{0:5s} R:{1:.4f} r100:{2:.4f} rtr{3:.2f} rval{4:.2f} rtest{5:.2f}'.format(str(k), total_reward / (k + 1), np.array(returns[-100:]).mean(),np.array(label[-100:]).mean(),r_val,r_test))