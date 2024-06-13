import sbibm
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from sbibm.algorithms import smc_abc, rej_abc, snpe, snle, snre
import pandas as pd
from sbibm.metrics import c2st,ksd,mmd,median_distance
import torch
import glob
import csv
import mlxp

#all tasks
print(sbibm.get_available_tasks())

def sample_posterior(cfg):
    """generate separated csv files with the posterior samples for each of the algos"""
    task = sbibm.get_task(cfg.model)  
    
    filename = f"{cfg.model}_{cfg.nb_obs}_obs_{cfg.n_sequential}_rounds_ntr_{cfg.n_train}_np_{cfg.n_posterior}_"

    # posterior_samples_abc, _, _ = rej_abc(task=task, num_samples=args.n_posterior, num_observation=args.nb_obs, num_simulations=args.n_train)
    # dataframe = pd.DataFrame(posterior_samples_abc.numpy())
    # dataframe.to_csv(filename+"posterior_samples_abc.csv")

    # posterior_samples_sabc, _, _ = smc_abc(task=task, num_samples=args.n_posterior, num_observation=args.nb_obs, num_simulations=args.n_train)
    # dataframe = pd.DataFrame(posterior_samples_sabc.numpy())
    # dataframe.to_csv(filename+"posterior_samples_sabc.csv")

    # posterior_samples_snle, _, _ = snle(task=task, neural_net=args.neural_net_snle, num_samples=args.n_posterior, num_observation=args.nb_obs, num_simulations=args.n_train, num_rounds=args.n_sequential)
    # dataframe = pd.DataFrame(posterior_samples_snle.numpy())
    # dataframe.to_csv(filename+"posterior_samples_snle-A"+'-'+args.neural_net_snle+".csv")

    # posterior_samples_snpe, _, _ = snpe(task=task,variant = args.variant_snpe,num_samples=args.n_posterior, num_observation=args.nb_obs, num_simulations=args.n_train, num_rounds=args.n_sequential)
    # dataframe = pd.DataFrame(posterior_samples_snpe.numpy())
    # dataframe.to_csv(filename+"posterior_samples_snpe-"+args.variant_snpe +'-'+args.neural_net_snpe+".csv")

    # posterior_samples_snre, _, _ = snre(task=task, variant=args.variant_snre, num_samples=args.n_posterior, num_observation=args.nb_obs, num_simulations=args.n_train, num_rounds = args.n_sequential)
    # dataframe = pd.DataFrame(posterior_samples_snre.numpy())
    # dataframe.to_csv(filename+"posterior_samples_snre-"+args.variant_snre +'-'+args.neural_net_snre+".csv")

    # posterior_samples_snpe, _, _ = snpe(task=task,variant = "A",num_samples=args.n_posterior, num_observation=args.nb_obs, num_simulations=args.n_train, num_rounds=args.n_sequential)
    # dataframe = pd.DataFrame(posterior_samples_snpe.numpy())
    # dataframe.to_csv(filename+"posterior_samples_snpe-A-mdn.csv")

    # posterior_samples_snre, _, _ = snre(task=task, variant="A", num_samples=args.n_posterior, num_observation=args.nb_obs, num_simulations=args.n_train, num_rounds = args.n_sequential)
    # dataframe = pd.DataFrame(posterior_samples_snre.numpy())
    # dataframe.to_csv(filename+"posterior_samples_snre-A"+'-'+args.neural_net_snre+".csv")

    # posterior_samples_snre, _, _ = snre(task=task, variant="C", num_samples=args.n_posterior, num_observation=args.nb_obs, num_simulations=args.n_train, num_rounds = args.n_sequential)
    # dataframe = pd.DataFrame(posterior_samples_snre.numpy())
    # dataframe.to_csv(filename+"posterior_samples_snre-C"+'-'+args.neural_net_snre+".csv")

    # posterior_samples_snre, _, _ = snre(task=task, variant="D", num_samples=args.n_posterior, num_observation=args.nb_obs, num_simulations=args.n_train, num_rounds = args.n_sequential)
    # dataframe = pd.DataFrame(posterior_samples_snre.numpy())
    # dataframe.to_csv(filename+"posterior_samples_snre-D"+'-'+args.neural_net_snre+".csv")

    #snle(task=task, neural_net=cfg.neural_net_snle, num_samples=cfg.n_posterior, num_observation=cfg.nb_obs, num_simulations=cfg.n_train, num_rounds=cfg.n_sequential)
    snpe(task=task,variant = "C",num_samples=cfg.n_posterior, num_observation=cfg.nb_obs, num_simulations=cfg.n_train, num_rounds=cfg.n_sequential, neural_net=cfg.neural_net_snpe)





def plots_per_rounds(n_sequential):
    """plot the posterior density for all algos for a given number of rounds"""
    plt.figure()
    filenames = sorted(glob.glob(f"*{n_sequential}_rounds*.csv"))

    end_n_obs = filenames[0].find("_obs")
    nb_obs = filenames[0][0:end_n_obs].split('_')[-1]
    model = filenames[0][0:filenames[0].find('_' + nb_obs)]
    task = sbibm.get_task(model)  
    reference_samples = task.get_reference_posterior_samples(num_observation=nb_obs)
    sns.kdeplot(reference_samples[:,0], label="ref")

    for i in range (len(filenames)):
        posterior_samples = pd.read_csv(filenames[i], index_col=0) 
        algo = (filenames[i].split('_')[-1]).split('.')[0]
        start = filenames[i].find('obs_')
        end = filenames[i].find('_rounds')
        n_sequential = filenames[i][start+4:end]
        sns.kdeplot(posterior_samples["0"], label=algo)
    plt.title(rf"$p(\theta | x_0)$ for {n_sequential} rounds")
    plt.legend()
    plt.savefig(f"Posteriors_" + filenames[0][0:filenames[0].find("_posterior")]+".png")

def plots_posteriors(rounds):
    """plot posterior density from csv files of posterior samples for all algos and given rounds 
    (place in the directory of the csv files)"""
    plt.figure(figsize=(20,20))
    plt.suptitle(r"$p(\theta | x_0)$", fontsize=25)
    
    for j in range(len(rounds)):
        plt.subplot(2,len(rounds)//2 + len(rounds)%2,j+1)
        #look for a given n_rounds
        filenames = sorted(glob.glob(f"*{rounds[j]}_rounds*.csv"))
        info= filenames[0][0:filenames[0].find("obs")+4]+filenames[0][filenames[0].find("ntr"):filenames[0].find("_posterior")]
        #search the n_obs in the csv title
        end_n_obs = filenames[0].find("_obs")
        nb_obs = filenames[0][0:end_n_obs].split('_')[-1]
        #search the model in the csv title
        model = filenames[0][0:filenames[0].find('_' + nb_obs)]
        task = sbibm.get_task(model)  
        reference_samples = task.get_reference_posterior_samples(num_observation=nb_obs)
        sns.kdeplot(reference_samples[:,0], label="ref")
        #plot all the algos
        for i in range (len(filenames)):
            posterior_samples = pd.read_csv(filenames[i], index_col=0) 
            algo = (filenames[i].split('_')[-1]).split('.')[0]
            start = filenames[i].find('obs_')
            end = filenames[i].find('_rounds')
            n_sequential = filenames[i][start+4:end]
            sns.kdeplot(posterior_samples["0"], label=algo)
        plt.title(f"{n_sequential} rounds")
        plt.legend()
    plt.subplots_adjust(left = 0.095, bottom=0.1, right = 0.95, top=0.9, hspace=0.15, wspace=0.15)
    plt.savefig(f"Posteriors_" + info +".png")

def pairplot(method,n_sequential,model):
    #filenames = glob.glob(f"{model}*{n_sequential}_rounds*{method}*.csv")
    filenames = ["gaussian_mixture_10000_posterior_samples_10_networks_mdn_2_rounds_5000_ntrain.csv"]
    for file in filenames :
        posterior_samples=pd.read_csv(file,index_col=0)
        end_n_obs = file.find("_obs")
        nb_obs = file[0:end_n_obs].split('_')[-1]
        task = sbibm.get_task(model)  
        reference_samples = task.get_reference_posterior_samples(num_observation=nb_obs)
        reference_samples=pd.DataFrame(reference_samples.numpy(),columns=[str(i) for i in range(len(posterior_samples.columns))])
        two_d_samples=[posterior_samples.assign(label=method),reference_samples.assign(label="ref")]
        two_d_samples=pd.concat(two_d_samples, ignore_index=True)
        
        size_subplot=len(posterior_samples.columns)
        plt.figure(figsize=(30+size_subplot,30+size_subplot))

        plt.suptitle(rf"$p(\theta | x_0)$ for {n_sequential} rounds",fontsize=35)
        for i in range(size_subplot):
            for j in range(i+1,size_subplot+1):
                plt.subplot(size_subplot,size_subplot,i*size_subplot+j)
                if j==i+1:
                    sns.kdeplot(posterior_samples[str(i)],label=method)
                    sns.kdeplot(reference_samples[str(i)],label="ref")
                    plt.legend(fontsize=20)
                    plt.xlabel(f"dim {j}",fontsize=20)
                    plt.ylabel("Density",fontsize=20)
                    
                else:
                    sns.kdeplot(data=two_d_samples, x=str(i),y=str(j-1),hue='label', common_norm = False)
                    plt.xlabel(f"dim {i+1}",fontsize=20)
                    plt.ylabel(f"dim {j}",fontsize=20)
                    #plt.title("2d density",fontsize=25)

        plt.subplots_adjust(left = 0.095, bottom=0.1, right = 0.95, top=0.9, hspace=0.3, wspace=0.3)
        plt.savefig(file.split('.')[0] + "_pairplot.png")

# def comparison_c2st_per_rounds(n_rounds):
#     """special function for c2st (too long otherwise) create a csv file with c2st scores 
#     (give the good list of files)"""

#     filenames = sorted(glob.glob(f"*{n_rounds}_rounds*.csv"))
#     end_n_obs = filenames[0].find("_obs")
#     nb_obs = filenames[0][0:end_n_obs].split('_')[-1]
#     model = filenames[0][0:filenames[0].find('_' + nb_obs)]
#     task = sbibm.get_task(model) 
#     reference_samples = task.get_reference_posterior_samples(num_observation=args.nb_obs)
#     c2st_dict = {}
#     for file in filenames:
    
#         algo = (file.split('_')[-1]).split('.')[0]
  
#         start = file.find('obs_')
#         end = file.find('_rounds')
#         n_sequential = file[start+4:end]
       
#         posterior_pred = pd.read_csv(file, index_col=0)
#         posterior_pred = torch.tensor(posterior_pred.values)

#         #accuracy = c2st(reference_samples, posterior_pred)
#         accuracy = torch.tensor([23])

#         c2st_dict[algo  + '_' + n_sequential + "_rounds"] = accuracy.item()
   
#     dataframe = pd.DataFrame(c2st_dict, index = [0])
#     dataframe.to_csv(f"c2st.csv")
#     return accuracy

def comparison_metric(rounds, metric, metric_name):
    """create a csv file for a given metric and given numbers of rounds"""

    filenames = sorted(glob.glob("*.csv"))
    #search the n_obs in the csv title
    end_n_obs = filenames[0].find("_obs")
    nb_obs = filenames[0][0:end_n_obs].split('_')[-1]
    #search the model in the csv title
    model = filenames[0][0:filenames[0].find('_' + nb_obs)]
    
    task = sbibm.get_task(model)  

    reference_samples = task.get_reference_posterior_samples(num_observation=args.nb_obs)
    algos = set()
    mmd_dict = [{"n_rounds":rounds[i]} for i in range(len(rounds))]
  
    for file in filenames:
        print("file", file)
        algo = (file.split('_')[-1]).split('.')[0]
        algos.add(algo)
        start = file.find('obs_')
        end = file.find('_rounds')
        n_sequential = int(file[start+4:end])
        posterior_pred = pd.read_csv(file, index_col=0)
        posterior_pred = torch.tensor(posterior_pred.values)
        accuracy = metric(reference_samples, posterior_pred).item()
        print("c2st", accuracy)
        for dict in mmd_dict:
      
            if n_sequential in dict.values():
                dict[algo] = accuracy
    labels = ["n_rounds"] + sorted(list(algos))
    info= filenames[0][0:filenames[0].find("obs")+4]+filenames[0][filenames[0].find("ntr"):filenames[0].find("_posterior")]

    with open(f"{metric_name}_{info}.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames = labels)
        writer.writeheader()
        for dict in mmd_dict:
            writer.writerow(dict)


def plot_comparison(filename):
    '''plots the scores for each algo and nb of rounds'''

    plt.figure(figsize=(10,10))
    dataframe = pd.read_csv(filename)
    n_rounds=dataframe['n_rounds']
    for i in range(1,len(dataframe.columns)):
        plt.plot(n_rounds,dataframe[dataframe.columns[i]],label=dataframe.columns[i],marker='o',linestyle='--')
    plt.xlabel('nb of rounds')
    plt.title(f"Comparison of {filename.split('_')[0]} scores")
    plt.legend()
    plt.savefig(f"{filename.split('.')[0]}_comparison.png")

@mlxp.launch(config_path='./configs')
def main(ctx: mlxp.Context):
    cfg = ctx.config

    #task = sbibm.get_task(args.model)  
    sample_posterior(cfg)
    #filenames = ["plots/gaussian_linear_1_obs_2_rounds_ntr_5000_np_10000_posterior_samples_abc.csv", "plots/gaussian_linear_1_obs_2_rounds_ntr_5000_np_10000_posterior_samples_snre.csv", "plots/gaussian_linear_1_obs_2_rounds_ntr_5000_np_10000_posterior_samples_snle.csv"]
    #filenames = sorted(glob.glob("*.csv"))
    # rounds = [1,2,5,10]
    # for r in rounds:
    #     plots_per_rounds(r)
    #rounds = [1,2,5,10]
    #plots_posteriors(rounds)
    #comparison_c2st(filenames, args, task)
    #plot_comparison('c2st_gaussian_mixture_nsf.csv')
    #comparison_metric([1,2,5,10], c2st, "c2st")
    #comparison_metric([5], c2st, "c2st")

    #pairplot("active",2,"gaussian_mixture")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='Simulator experience'
    # )
    # parser.add_argument('--model', '-m', type=str, default="gaussian_linear",
    #                     help="The model to use among : 'gaussian_linear', 'bernoulli_glm', 'slcp', 'two_moons', 'gaussian_linear_uniform', 'gaussian_mixture', 'slcp_distractors', 'bernoulli_glm_raw'")
    # parser.add_argument('--snre_type', '-snre_t', type=str, default="B",
    #                     help='The type of SNRE A or B')
    # parser.add_argument('--n_train', '-ntr', type=int, default=5000,
    #                     help='Number of train samples to make')
    # parser.add_argument('--n_sequential', '-ns', type=int, default=1,
    #                     help='Number of rounds to do in sequential algo')
    # parser.add_argument('--nb_obs', '-nobs', type=int, default=1,
    #                     help='Number of observations x to have')
    # parser.add_argument('--neural_net_snpe', '-nnsnpe', type=str, default="nsf",
    #                     help='The neural network for the posterior estimator among maf / mdn / made / nsf')
    # parser.add_argument('--variant_snpe', '-vsnpe', type=str, default="C",
    #                     help="The variant of SNPE among 'A' or 'C'")
    # parser.add_argument('--variant_snre', '-vsnre', type=str, default="B",
    #                     help="The variant of SNRE among 'A', 'B', 'C' or 'D' for BNRE")
    # parser.add_argument('--neural_net_snle', '-nnsnle', type=str, default="maf",
    #                     help='The neural network for the likelihood estimator among maf / mdn / made / nsf')
    # parser.add_argument('--neural_net_snre', '-nnsnre', type=str, default="resnet",
    #                     help='The neural network for the ratio estimator among linear / mlp / resnet')
    # parser.add_argument('--batch_size_simulator', '-bss', type=int, default=1000,
    #                     help='Bacth size for the simulatior')
    # parser.add_argument('--batch_size_training', '-bstr', type=int, default=10000,
    #                     help='Bacth size for the simulator')
    # parser.add_argument('--n_posterior', '-np', type=int, default=10000,
    #                     help='Number of posterior samples theta')

    # args = parser.parse_args()
    main()
    
