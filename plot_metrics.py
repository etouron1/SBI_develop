from sbi.utils.get_nn_models import posterior_nn
import torch
from sbi import inference as inference
import matplotlib.pyplot as plt

import pandas as pd
import math
import csv
import sbibm

from sbibm.algorithms.sbi.snpe import wrap_posterior
from sbibm.metrics import c2st, mmd

import glob

import yaml
import json
from torch import nn
from matplotlib import collections  as mc

def plot_accuracy(filename, metric):
    plt.figure(figsize=(20,20))
    #color = ['C0', 'C1', 'blue', 'green', 'pink']
    #color = ['1F77B4', 'FF7F0E', '2CA02C', 'D62728']#, '9467BD', '8C564B', 'E377C2', '7F7F7F', 'BCBD22', '17BECF']
    colors = ['firebrick', 'darkolivegreen','darkorange', 'green', 'turquoise', 'blueviolet', 'y', 'k']
    #label = [r"$\frac{1}{2}||\nabla_{\theta} log(\tilde{q}_{\Delta}(\theta_n|x_n)) - \nabla_{\theta} log(\tilde{p}(\theta_n|x_n))||^2$", r"\Delta_{\theta} log(\tilde{q}_{\Delta}(\theta_n|x_n)) + \frac{1}{2} || \nabla_{\theta} log(\tilde{q}_{\Delta}(\theta_n|x_n)) || ^2$"]
    label = ["loss 1", "loss 2"]
    for i in range(len(filename)):
        df = pd.read_csv(filename[i])
        print(df)
        y_min = (df[metric]-df['min_'+ metric]).to_list()
        #y_min = df['min_'+metric].to_list()

        y_max = (df['max_'+ metric]-df[metric]).to_list()
        #y_max = df['max_'+ metric].to_list()

        #y_error = torch.transpose(torch.cat((y_min, y_max),dim=1),0,1)
        y_error = [y_min, y_max]
        lines = []
        
        for j in range (10):
            deb = df['min_'+metric][j]
            fin = df['max_'+metric][j]
    
            plt.vlines(x=j+1, ymin=deb, ymax=fin, linestyle='dashed', color=colors[i])

        plt.plot(df['round'], df[metric], linestyle='solid', marker='o', label=label[i], linewidth=2, markersize=10, color=colors[i])
     
        plt.scatter(df['round'], df["max_"+metric],color=colors[i])
        plt.scatter(df['round'], df["min_"+metric],color=colors[i])

        #plt.errorbar(df['round'], df[metric], yerr=y_error, fmt='.k', ecolor='light'+color[i], capsize=2)
    plt.ylabel(metric, fontsize=20)
    plt.xlabel('round', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.title(metric+' scores', fontsize=30)
    plt.suptitle("Gaussian Linear-NSF", fontsize=40)
    #plt.suptitle("Two Moons", fontsize=40)

    plt.legend(fontsize=30)
    #plt.show()
    plt.savefig(filename[0][:-4]+'_'+metric+'.png')



def main():
    plot_accuracy(["snpe_d_means_nsf_diff_grad.csv", "snpe_d_means_nsf_laplacian.csv"], "c2st")
    
if __name__ == "__main__":
    main()
    #compute_mean_seed([0,10,20],"two_moons_10_networks_nsf_10_rounds_prop_0.75")
    #plot_accuracy(filenames, 'mmd')
    # plot_accuracy(filenames, 'c2st')



