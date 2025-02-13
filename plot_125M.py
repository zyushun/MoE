import json
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import re


def smooth(res,rate=0.6):
    s = -1
    res1 = []
    for p in res:
        if (s<0):
            s = p
        else:
            s = rate*s + (1-rate)*p
        res1.append(s)
    return res1



def extract_val(filename):
    # val loss
    steps = []
    loss = [10]
    i = 0
    with open(filename,"r") as f:
        for line in f.readlines():
            if(line.find("val_loss")>0):
                a = re.split(":| |,|\n",line)
                print('a', a)
                loss.append(float(a[6]))

            else:
                continue

            print('loss', loss[-1])
    return loss
    # return loss

def extract_train(filename):
    # this is for train loss
    steps = []
    loss = [10]
    i = 0
    with open(filename,"r") as f:
        for line in f.readlines():
            if(line.find("val")>0):
                continue
            if(line.find("loss")>0):
                i = i+1
                if (i % 1 ==1):
                    continue
                a = re.split(":| |,|\n",line)
                #print('a', a)
                
                # print('a', a[21])
                #steps.append(int(a[16]))
                if a[16] != '':
                    #print('a[16]', a[16])
                    loss.append(float(a[16]))
                elif a[17] != '': 
                    #print('a[17]', a[17])
                    loss.append(float(a[17]))
            print('loss', loss[-1])
    loss = loss[:10000]
    #loss = loss[::100]
    #loss = smooth(loss, 0.2)
    loss = loss[::10]
    #loss = smooth(loss, 0.6)
    loss = smooth(loss, 0.2)
    return loss


# loss_list_muon = extract_val('out_125M/191827.out')   

loss_list_adamw = extract_val('out_125M/209024.out')   
loss_list_mini = extract_val('out_125M/209032.out')   
loss_list_muon = extract_val('out_125M/209038.out')   
loss_list_mini_empty_v = extract_val('out_125M/209053.out')   

loss_list_muon_ours = extract_val('out_125M/214601.out')   # 0.0018

loss_list_muon_ours_05 = extract_val('out_125M/214602.out')    #0.05

loss_list_muon_ours_05_4optimizers = extract_val('out_125M/214630.out')    #0.05


# loss_list_muon_no_momentum_warmup = extract_val('out_125M/214655.out')    #0.05


loss_list_muon_ours_qkv_head_inverse = extract_val('out_125M/214673.out')    #0.05

loss_list_muon_ours_qkv_head_inverse_no_biascorrection = extract_val('out_125M/214675.out')    #0.05



plt.rcParams["axes.autolimit_mode"] = "round_numbers"
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["legend.fontsize"] = 15#25 #20
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = True
plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams["xtick.labelsize"] = 20#20
plt.rcParams["ytick.labelsize"] = 20#20
plt.rcParams["lines.markersize"] = 10
plt.rcParams["font.family"] = "serif"



plt.rcParams["figure.figsize"] = (12, 6)




iteration = [i * 125 for i in range(len(loss_list_adamw))]


linewidth = 1.5#1.25

# train loss


plt.figure()


# mini

plt.plot(iteration, loss_list_adamw, label = 'AdamW-0.0018',  linewidth = linewidth, linestyle = '-', alpha = 1, color = 'blue')

plt.plot(iteration, loss_list_mini, label = 'Adam-mini-0.0018',  linewidth = linewidth, linestyle = '--', alpha = 1, color = 'red')


# plt.plot(iteration, loss_list_mini_empty_v, label = 'Adam-mini-empty-v-0.0018',  linewidth = linewidth, linestyle = '--', alpha = 1, color = 'orange')

plt.plot(iteration, loss_list_muon, label = 'Muon-0.05',  linewidth = linewidth, linestyle = '--', alpha = 1, color = 'green')


plt.plot(iteration[:len(loss_list_muon_ours)], loss_list_muon_ours, label = 'Muon-ours-0.0018',  linewidth = linewidth, linestyle = '--', alpha = 1, color = 'orange')

plt.plot(iteration[:len(loss_list_muon_ours_05)], loss_list_muon_ours_05, label = 'Muon-ours-0.05',  linewidth = linewidth, linestyle = '--', alpha = 1)

plt.plot(iteration[:len(loss_list_muon_ours_05_4optimizers)], loss_list_muon_ours_05_4optimizers, label = 'Muon-ours-0.05-4optimizers',  linewidth = linewidth, linestyle = '--', alpha = 1)

# plt.plot(iteration[:len(loss_list_muon_no_momentum_warmup)], loss_list_muon_no_momentum_warmup, label = 'Muon-no-momentum-warmup-0.05',  linewidth = linewidth, linestyle = '--', alpha = 1, color = 'black')


plt.plot(iteration[:len(loss_list_muon_ours_qkv_head_inverse)], loss_list_muon_ours_qkv_head_inverse, label = 'Muon-ours-qkv-head-inverse-0.05',  linewidth = linewidth, linestyle = '--', alpha = 1, color = 'purple')

# plt.plot(iteration[:len(loss_list_muon_ours_qkv_head_inverse_no_biascorrection)], loss_list_muon_ours_qkv_head_inverse_no_biascorrection, label = 'Muon-ours-qkv-head-inverse-no-biascorrection-0.05',  linewidth = linewidth, linestyle = '--', alpha = 1, color = 'pink')


# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Iteration', fontname='Serif')
# plt.ylabel('Validation loss')
plt.ylabel('Val loss', fontname='Serif')
plt.ylim([2.8,8.5])
plt.title('GPT-2-125M Pretrain')
# plt.ylim([2.2,4.5])
# plt.xlim([1e1, 500])
plt.legend()
# plt.title(f'Llama-1B Pre-training')
plt.savefig(f'figures/0106_125M_valloss.png', bbox_inches='tight')
plt.close()
