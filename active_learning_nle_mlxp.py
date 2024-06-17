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
from sbibm.metrics import c2st
import csv
import mlxp
import yaml


def build_networks_ensemble(nb_networks, neural_net= "made",hidden_features = 50,z_score_x= "independent",z_score_theta = "independent",):
    networks_ensemble = []
    for _ in range(nb_networks):
        # mdn = build_mdn(batch_x=batch_x,batch_y=batch_theta, num_components=num_components)
        # networks_ensemble.append(mdn)
        density_estimator_fun = likelihood_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
        #sortie de mean layer pour mdn
        num_components = 10,
        #nb de bloc pour maf et nsf
        num_transforms = 5,
        #out features pour les splines pour nsf
        num_bins= 10,)
        networks_ensemble.append(density_estimator_fun)
    return networks_ensemble


def acquisition_theta(theta, x_0, density_estimators, prior):
    # print(x_0.size())
    # theta = torch.from_numpy(theta).unsqueeze(0)[0]
    # print(theta)
    mean = 0
    likelihoods =torch.zeros(1)
    for est in density_estimators:
        likelihood = torch.exp(est.log_prob(x_0, context=theta))
        likelihoods=torch.cat((likelihoods, likelihood), dim=0)
        mean +=likelihood/len(density_estimators)
    
    #print("l", (likelihoods-[mean for _ in range (len(density_estimators))])**2)
    var_likelihood = torch.mean((likelihoods - mean)**2)
    log_prior = prior.log_prob(theta)
    # print("log_prior", log_prior)
    # print("logvar", torch.log(var_likelihood))
    return log_prior + 0.5*torch.log(var_likelihood)

     
#      loss = log_prior(theta) + log_likelihood
def plot_acquisition(x_0, density_estimators,prior,fig):
    n_samples = 500
    thetas = prior.sample((n_samples,)).reshape(n_samples,1,2)
    y=[]
    for t in thetas:
        y.append(-acquisition_theta(t, x_0, density_estimators, prior).item())

    df = pd.DataFrame.from_dict({"x":thetas[:,:,0].squeeze(), "y" : thetas[:,:,1].squeeze(), "data":y})
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    plot = ax.plot_trisurf(df["x"], df["y"], df["data"], cmap=plt.cm.viridis, alpha=0.4)
    fig.colorbar(plot)
    return ax
    #sns.kdeplot(data=df, x="x", y="y", fill=True, cbar=True)
   
    #plt.show()


#neural_net = likelihood_nn(model="mdn")

#x_0 = torch.tensor([[1.0, 1.0]])
# x_0 = task.get_observation(num_observation=1).reshape(1,2)
# num_simulations = 12
# #prior = torch.distributions.MultivariateNormal(loc= torch.zeros(10), precision_matrix= torch.eye(10))
# prior = task.get_prior_dist()
# simulator = task.get_simulator(max_calls=num_simulations)
# simulation_batch_size = 1000
# training_batch_size = 2
# num_rounds = 2
# num_simulations_per_round = math.floor(num_simulations / num_rounds)
# max_num_epochs = 2**31 - 1
# prop_new_theta = 0.5
# nb_new_theta = int(num_simulations_per_round*prop_new_theta)
# simulator_new = task.get_simulator(max_calls=nb_new_theta*(num_rounds-1))
# num_posterior_samples = 1000
# num_posterior_samples_per_network = int(num_posterior_samples/nb_networks)

def train_networks(networks_ensemble, task, num_simulations, num_rounds, simulation_batch_size, training_batch_size,prop_new_theta, neural_net_name,model,logger):
    x_0 = task.get_observation(num_observation=1).reshape(1,2)
    prior = task.get_prior_dist()
    num_simulations_per_round = math.floor(num_simulations / num_rounds)
    simulator = task.get_simulator(max_calls=num_simulations)
    max_num_epochs = 2**31 - 1
    nb_new_theta = int(num_simulations_per_round*prop_new_theta)
    simulator_new = task.get_simulator(max_calls=nb_new_theta*(num_rounds-1))

    #mm inference pour tous  les networks
    inference_methods = []
    #file = open("trace.txt", "a")
    for i in range (len(networks_ensemble)):
        inference_methods.append(inference.SNLE_A(density_estimator=networks_ensemble[i],prior=prior,))

    if simulation_batch_size > num_simulations_per_round:
            simulation_batch_size = num_simulations_per_round
            print("Reduced simulation_batch_size to num_simulation_per_round")

    if training_batch_size > num_simulations_per_round:
            training_batch_size = num_simulations_per_round
            print("Reduced training_batch_size to num_simulation_per_round")
            
    for r in range(num_rounds):
        if r==0:

            print(f"################ Simulate theta,x for round {r+1} ################")
            proposal = prior
            theta, x = inference.simulate_for_sbi(
                    simulator,
                    proposal,
                    num_simulations=num_simulations_per_round,
                    simulation_batch_size=simulation_batch_size,
                )
 
        density_estimators = []
        for i in range (len(networks_ensemble)):
            print(f"################ Network {i+1}/{len(networks_ensemble)} ################")

            density_estimator = inference_methods[i].append_simulations(
                theta, x, from_round=r
            ).train(
                training_batch_size=training_batch_size,
                retrain_from_scratch=False,
                discard_prior_samples=False,
                show_train_summary=True,
                max_num_epochs=max_num_epochs,
            )

            density_estimators.append(density_estimator)
            
            #torch.save(density_estimator.state_dict(),f"state_dict_network_{i+1}_round_{r+1}.pt")
            logger.log_checkpoint({'state_dict':density_estimator.state_dict()}, log_name=f"state_dict_network_{i+1}_round_{r+1}")
        
        if r < num_rounds-1:
            fig = plt.figure(figsize=(30,30))
            ax = plot_acquisition(x_0,  density_estimators, prior, fig)
            print("################ Find new theta for MaxVar ################")
            
            loss_pred = torch.tensor([float("Inf")])
            loss = torch.tensor([-float("Inf")])
            while loss.item() == -float("Inf") or loss.item() == float("Inf") or math.isnan(loss.item()):
                theta_init = prior.sample().reshape(1,2)
    
                theta_init.requires_grad = True

                optimizer = torch.optim.Adam([theta_init], lr=0.1)
                max_epochs = 200
                epoch = 0
                loss = torch.tensor([-float("Inf")])
                loss_pred = torch.tensor([float("Inf")])
                trajectory = theta_init.detach()
                while epoch < max_epochs and loss < loss_pred:
                    
                    if epoch >0:
                        loss_pred = loss

                    loss = -acquisition_theta(theta_init, x_0, density_estimators, prior)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #val_loss.append(loss.item())
                
                    epoch +=1
                    #print(epoch)
                    #print("init", theta_init)
                    trajectory=torch.cat((trajectory, theta_init.detach()), dim=0)
                # if k==1:
                #     loss =torch.tensor([float("Inf")]) 
                #     k=2   
                
    
                print(f"Converged at epoch {epoch}/{max_epochs} with a loss {loss.item()}") 
                logger.log_metrics({'loss': loss.item(),'round': r+1}, log_name='acquisition_loss')  
                #file.write(f"\nRound : {r+1} -- Converged at epoch {epoch}/{max_epochs} with a loss {loss.item()}\n")
            #ax.scatter(theta_init[:,0].item(), theta_init[:,1].item(), loss.item(), color="blue")
        
            ax.scatter(trajectory[:,0], trajectory[:,1], loss.item(), color="blue")
                
            print(f"################ Neighborhood of theta MaxVar for round {r+1} ################")

            new_theta = theta_init
            dim = theta_init.size(1)
            sigma = 0.01
            noise = torch.distributions.MultivariateNormal(torch.zeros(dim), sigma*torch.eye(dim))

            new_thetas = new_theta + noise.sample((nb_new_theta-1,))
            new_thetas = torch.cat((new_theta, new_thetas), dim=0)
      
            for i in range (nb_new_theta):
                ax.scatter(new_thetas[i,0].item(), new_thetas[i,1].item(), -acquisition_theta(new_thetas[i].reshape(1,2), x_0, density_estimators, prior).item(), marker="+", color="red")
      
            plt.savefig(f"{model}_theta_maxvar_for_round_{r+2}_{len(networks_ensemble)}_networks_{neural_net_name}.png")
            #logger.log_artifacts({'plot_theta': fig,'round':r+1}, artifact_name='theta_max_var', artifact_format='image')
            theta, x = inference.simulate_for_sbi(
                        simulator,
                        proposal,
                        num_simulations=int(num_simulations_per_round*(1-prop_new_theta)),
                        simulation_batch_size=simulation_batch_size,
                    )
            theta=torch.cat((theta, new_thetas), dim=0).detach()
            new_x = simulator_new(new_thetas)
            x = torch.cat((x, new_x), dim=0).detach()
            #file.close()
    return density_estimators

def posterior_sampling(task, nb_networks, density_estimators, num_posterior_samples, round, num_rounds, num_simulations, neural_net_name, model):

    automatic_transforms_enabled = True
    mcmc_method= "slice_np_vectorized"
    mcmc_parameters= {
            "num_chains": 100,
            "thin": 10,
            "warmup_steps": 25,
            "init_strategy": "resample",
            "init_strategy_parameters": {"num_candidate_samples": 10000,}
        }
    x_0 = task.get_observation(num_observation=1).reshape(1,2)
    prior = task.get_prior_dist()
    num_posterior_samples_per_network = int(num_posterior_samples/nb_networks)
    for i in range (nb_networks):
   
        (potential_fn,theta_transform,) = inference.likelihood_estimator_based_potential(
            density_estimators[i],prior,x_0,
            enable_transform=not automatic_transforms_enabled,)
        
        posterior = inference.MCMCPosterior(
                    potential_fn=potential_fn,
                    proposal=prior,  # proposal for init_strategy
                    theta_transform=theta_transform,
                    method=mcmc_method,
                    **mcmc_parameters,
                )
        if i==0:
            posterior_samples = posterior.sample((num_posterior_samples_per_network,))
        else:
            posterior_samples=torch.cat((posterior_samples, posterior.sample((num_posterior_samples_per_network,))), dim=0)

    posterior_samples = pd.DataFrame(posterior_samples.detach().numpy(),columns=[str(i) for i in range(posterior_samples.size(1))])
    reference_samples = task.get_reference_posterior_samples(num_observation=1)

    reference_samples=pd.DataFrame(reference_samples.numpy(),columns=[str(i) for i in range(reference_samples.size(1))])
    two_d_samples=[posterior_samples.assign(label="active"),reference_samples.assign(label="ref")]
    two_d_samples=pd.concat(two_d_samples, ignore_index=True)
    filename = f"{model}_{num_posterior_samples}_posterior_samples_{nb_networks}_networks_{neural_net_name}_{num_rounds}_rounds_{num_simulations}_ntrain"
    two_d_samples.to_csv(filename + ".csv")
    plt.figure(figsize=(30,30))
    plt.subplot(121)
    plt.title("active")
    sns.kdeplot(data = posterior_samples, x="0", y="1")
    plt.subplot(122)
    plt.title("ref")
    sns.kdeplot(data = reference_samples, x="0", y="1")
    #plt.show()
    plt.savefig(filename + ".png")


def posterior_sampling_per_round(task, round, nb_networks, density_estimators, num_posterior_samples, filename,job_id):
    with open(filename, "w",newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["0", "1", "label"])
                file.close()

    automatic_transforms_enabled = True
    mcmc_method= "slice_np_vectorized"
    mcmc_parameters= {
            "num_chains": 100,
            "thin": 10,
            "warmup_steps": 25,
            "init_strategy": "resample",
            "init_strategy_parameters": {"num_candidate_samples": 10000,}
        }
    x_0 = task.get_observation(num_observation=1).reshape(1,2)
    prior = task.get_prior_dist()
    num_posterior_samples_per_network = int(num_posterior_samples/nb_networks)

    for i in range (nb_networks):
        model_file=pd.read_pickle(f'logs/{job_id}/artifacts/pickle/state_dict_network_{i+1}_round_{round+1}')
        density_estimators[i].load_state_dict(model_file['state_dict'])
        #density_estimators[i].load_state_dict(torch.load(f"state_dict_network_{i+1}_round_{round+1}.pt"))
   
        (potential_fn,theta_transform,) = inference.likelihood_estimator_based_potential(
            density_estimators[i],prior,x_0,
            enable_transform=not automatic_transforms_enabled,)
        
        posterior = inference.MCMCPosterior(
                    potential_fn=potential_fn,
                    proposal=prior,  # proposal for init_strategy
                    theta_transform=theta_transform,
                    method=mcmc_method,
                    num_workers= 10,
                    **mcmc_parameters,
                )
        posterior_samples = posterior.sample((num_posterior_samples_per_network,))
        posterior_samples = pd.DataFrame(posterior_samples.detach().numpy(),columns=["0", "1"])
        posterior_samples=posterior_samples.assign(label="active")
        posterior_samples.to_csv(filename, mode="a", header=False, index=False)

def superpose_posterior_plot(file, n_sequential, nb_networks, neural_net):
    data = pd.read_csv(file, index_col=0)
    posterior_samples = data[data["label"]=="active"][['0', '1']]
 
    reference_samples = data[data["label"]=="ref"][['0', '1']]
    size_subplot=len(posterior_samples.columns)
    plt.figure(figsize=(30+size_subplot,30+size_subplot))

    plt.suptitle(rf"$p(\theta | x_0)$ for {n_sequential} rounds and {nb_networks} networks-{neural_net}",fontsize=35)
    for i in range(size_subplot):
        for j in range(i+1,size_subplot+1):
            plt.subplot(size_subplot,size_subplot,i*size_subplot+j)
            if j==i+1:
                sns.kdeplot(posterior_samples[str(i)],label="active")
                sns.kdeplot(reference_samples[str(i)],label="ref")
                plt.legend(fontsize=20)
                plt.xlabel(f"dim {j}",fontsize=20)
                plt.ylabel("Density",fontsize=20)
                
            else:
                sns.kdeplot(data=data, x="0",y="1",hue='label', common_norm = False)
                plt.xlabel(f"dim {i+1}",fontsize=20)
                plt.ylabel(f"dim {j}",fontsize=20)
                #plt.title("2d density",fontsize=25)

    plt.subplots_adjust(left = 0.095, bottom=0.1, right = 0.95, top=0.9, hspace=0.3, wspace=0.3)
    plt.savefig(file.split('.')[0] + "_pairplot.png")


def c2st_comparison(filename):
    samples = pd.read_csv(filename, index_col=0)
    posterior_pred = samples[samples["label"]=="active"]
 
    posterior_pred = torch.tensor(posterior_pred[['0', '1']].values)
  
    reference_samples = samples[samples["label"]=="ref"]
    reference_samples = torch.tensor(reference_samples[['0', '1']].values)


    accuracy = c2st(reference_samples, posterior_pred).item()
    print(accuracy)


    
@mlxp.launch(config_path='./configs/')
def main(ctx: mlxp.Context):

    cfg = ctx.config
    logger = ctx.logger

    if cfg.phase=="train":
        networks_ensemble = build_networks_ensemble(cfg.nb_networks,  neural_net=cfg.neural_net)
        task = sbibm.get_task(cfg.model)
        
        density_estimators = train_networks(networks_ensemble, task, num_simulations=cfg.nb_train, num_rounds=cfg.nb_rounds, 
                    simulation_batch_size=cfg.batch_size_simulator, training_batch_size=cfg.batch_size_training,
                    prop_new_theta=cfg.prop_new_theta, neural_net_name=cfg.neural_net, model=cfg.model,logger=logger)
    
    if cfg.phase=="sample":
        with open(f"logs/{cfg.job_id}/metadata/config.yaml", "r") as f:
            info= yaml.safe_load(f)
        model=info['model']
        task = sbibm.get_task(model)
        prior = task.get_prior_dist()
        batch_theta = prior.sample((10,))
        simu= task.get_simulator()
        batch_x = simu(batch_theta)
        networks_ensemble = build_networks_ensemble(nb_networks=info['nb_networks'],  neural_net=info['neural_net'])
        density_estimators = []

        for i in range (len(networks_ensemble)):
            inference_method=inference.SNLE_A(density_estimator=networks_ensemble[i],prior=prior,)
            density_estimators.append(inference_method.append_simulations(
                    batch_theta, batch_x).train(training_batch_size=100,max_num_epochs=5))
            # for name, param in density_estimators[i].named_parameters():
            #     print(name, param)
    

        for round in range(info['nb_rounds']):
            
            filename = f"{model}_{cfg.nb_posterior}_posterior_samples_{info['nb_networks']}_networks_{info['neural_net']}_{round+1}_rounds_{info['nb_train']}_ntrain_prop_{info['prop_new_theta']}.csv"
            posterior_sampling_per_round(task=task, round = round,nb_networks=info['nb_networks'], density_estimators=density_estimators, num_posterior_samples=cfg.nb_posterior, filename=filename,job_id=cfg.job_id)
        

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='Simulator experience'
    # )
    # parser.add_argument('--phase', '-p', type=str, default="train",
    #                     help='Choose to "train" or "sample"')
    # parser.add_argument('--model', '-m', type=str, default="gaussian_mixture",
    #                     help="The model to use among : 'gaussian_linear', 'bernoulli_glm', 'slcp', 'two_moons', 'gaussian_linear_uniform', 'gaussian_mixture', 'slcp_distractors', 'bernoulli_glm_raw'")
    # parser.add_argument('--nb_train', '-ntr', type=int, default=5000,
    #                     help='Number of train samples to make')
    # parser.add_argument('--nb_rounds', '-nr', type=int, default=1,
    #                     help='Number of rounds to do in sequential algo')
    # parser.add_argument('--nb_obs', '-nobs', type=int, default=1,
    #                     help='Number of observations x to have')
    # parser.add_argument('--batch_size_simulator', '-bss', type=int, default=1000,
    #                     help='Bacth size for the simulatior')
    # parser.add_argument('--batch_size_training', '-bstr', type=int, default=10000,
    #                     help='Bacth size for the simulator')
    # parser.add_argument('--nb_posterior', '-np', type=int, default=10000,
    #                     help='Number of posterior samples theta')
    # parser.add_argument('--nb_networks', '-nn', type=int, default=1,
    #                     help='Number of networks to have')
    # parser.add_argument('--prop_new_theta', '-prop', type=float, default=0.2,
    #                     help='Proportion of thetas from MaxVar in the new batch')
    # parser.add_argument('--neural_net', '-net', type=str, default="mdn",
    #                     help='The neural network for the estimator among maf / mdn / made / nsf')
    # args = parser.parse_args()
    main()
    # superpose_posterior_plot("gaussian_mixture_10000_posterior_samples_10_networks_nsf_2_rounds_5000_ntrain.csv", 2,  10, "nsf")
    # c2st_comparison("gaussian_mixture_10000_posterior_samples_10_networks_nsf_2_rounds_5000_ntrain.csv")
    
