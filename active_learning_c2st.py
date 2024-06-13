from sbi.inference.snpe.snpe_a import SNPE_A_MDN
from sbi import inference as inference
import sbibm
from sbi.utils.get_nn_models import classifier_nn, likelihood_nn, posterior_nn
from sbi.neural_nets.mdn import build_mdn
import torch
import math
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import argparse
from sbibm.metrics import c2st, mmd
import csv
import glob


def c2st_comparison(filename, metric):
    
    task = sbibm.get_task('gaussian_mixture')
 
    # samples = pd.read_csv(filename, index_col=0,header=None)
    print(filename)
    samples = pd.read_csv(filename, index_col=False)
    #samples.columns = ['0', '1','label']
    #posterior_pred = samples[samples["label"]=="active"]
    posterior_pred = torch.tensor(samples[[str(i) for i in range(2)]].values)
    #print(posterior_pred)
    # reference_samples = samples[samples["label"]=="ref"]
    # reference_samples = torch.tensor(reference_samples[['0', '1']].values)
    reference_samples = task.get_reference_posterior_samples(num_observation=10)

    accuracy = metric(reference_samples, posterior_pred).item()
    #accuracy = mmd(reference_samples, posterior_pred).item()
    print(accuracy)
    return(accuracy)

    

def main(args):

    # networks_ensemble = build_networks_ensemble(args.nb_networks,  neural_net=args.neural_net)
  
    # task = sbibm.get_task(args.model)
    
    # train_sample_networks(networks_ensemble, task, num_simulations=args.nb_train, num_rounds=args.nb_rounds, num_posterior_samples=args.nb_posterior,
                #    simulation_batch_size=args.batch_size_simulator, training_batch_size=args.batch_size_training,
                #    prop_new_theta=args.prop_new_theta, neural_net_name=args.neural_net, model=args.model)
    filenames = sorted(glob.glob("*.csv"))
    #print(filenames[1:])
    with open("accuracy.csv", "w") as f:
        writer = csv.writer(f)
        for file in filenames:
            #print(file)
            num_round=file[file.find("maf")+4:file.find('round')-1]
            #print(num_round)
            #accuracy[file]=c2st_comparison("essai_c2st.csv")
            accuracy=c2st_comparison(file, c2st)
            writer.writerow([num_round, accuracy])
    
    with open("mmd.csv", "w") as f:
        writer = csv.writer(f)
        for file in filenames:
            #print(file)
            if file != "accuracy.csv":
                num_round=file[file.find("maf")+4:file.find('round')-1]
                #print(num_round)
                #accuracy[file]=c2st_comparison("essai_c2st.csv")
                accuracy=c2st_comparison(file, mmd)
                writer.writerow([num_round, accuracy])

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simulator experience'
    )
    parser.add_argument('--model', '-m', type=str, default="gaussian_mixture",
                        help="The model to use among : 'gaussian_linear', 'bernoulli_glm', 'slcp', 'two_moons', 'gaussian_linear_uniform', 'gaussian_mixture', 'slcp_distractors', 'bernoulli_glm_raw'")
    parser.add_argument('--nb_train', '-ntr', type=int, default=5000,
                        help='Number of train samples to make')
    parser.add_argument('--nb_rounds', '-nr', type=int, default=1,
                        help='Number of rounds to do in sequential algo')
    parser.add_argument('--nb_obs', '-nobs', type=int, default=1,
                        help='Number of observations x to have')
    parser.add_argument('--batch_size_simulator', '-bss', type=int, default=1000,
                        help='Bacth size for the simulatior')
    parser.add_argument('--batch_size_training', '-bstr', type=int, default=10000,
                        help='Bacth size for the simulator')
    parser.add_argument('--nb_posterior', '-np', type=int, default=10000,
                        help='Number of posterior samples theta')
    parser.add_argument('--nb_networks', '-nn', type=int, default=1,
                        help='Number of networks to have')
    parser.add_argument('--prop_new_theta', '-prop', type=float, default=0.2,
                        help='Proportion of thetas from MaxVar in the new batch')
    parser.add_argument('--neural_net', '-net', type=str, default="mdn",
                        help='The neural network for the estimator among maf / mdn / made / nsf')
    args = parser.parse_args()
    main(args)
    #superpose_posterior_plot("gaussian_mixture_10000_posterior_samples_10_networks_nsf_2_rounds_5000_ntrain.csv", 2,  10, "nsf")
    #c2st_comparison("gaussian_mixture_10000_posterior_samples_10_networks_nsf_2_rounds_5000_ntrain.csv")
    