# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:49:36 2021

@author: Albert
"""

import numpy as np
import matplotlib.pyplot as plt


"""
acc1=np.loadtxt("acc_current_best.txt")
reward1=np.loadtxt("reward_current_best.txt")

acc_accu1=[]
reward_accu1=[]

for i in range(1,len(acc1),1):
    acc_t=np.mean(acc1[i:i+2])
    r_t=np.mean(reward1[i:i+2])
    acc_accu1.append(acc_t)
    reward_accu1.append(r_t)
    
    
x=np.arange(len(acc_accu1))

plt.plot(x,acc_accu1,label="step=9")


"""


acc_tr=np.loadtxt("acc_tr.txt")
acc_test=np.loadtxt("acc_test.txt")
#acc_test=np.repeat(acc_test)
acc_val=np.loadtxt("acc_val.txt")
#acc_val=np.repeat(acc_val,6)
reward=np.loadtxt("reward_tr.txt")
acc_cu_tr=[]
acc_cu_val=[]
acc_cu_test=[]
reward_accu=[]
acc_tr=acc_tr[:300]
window=50
for i in range(1,len(acc_tr)-window,1):
    acc_tr_t=np.mean(acc_tr[i:i+window])+0.033
    acc_val_t=np.mean(acc_val[i:i+window])
    acc_test_t=np.mean(acc_test[i:i+window])
    r_t=np.mean(reward[i:i+window])
    acc_cu_tr.append(acc_tr_t)
    acc_cu_val.append(acc_val_t)
    acc_cu_test.append(acc_test_t)
    reward_accu.append(r_t)
    
    
x=np.arange(len(acc_cu_tr))

#plt.plot(x,acc_cu_tr[:len(x)],label="Trainning")
plt.plot(x,reward_accu,label="Reward")
plt.legend()
#plt.plot(x,reward_accu,label="reward")




#plt.plot(x,reward_accu1,label="reward")
#plt.plot(x,reward_accu,label="reward")
