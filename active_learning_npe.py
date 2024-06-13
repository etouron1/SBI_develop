from sbi.utils.get_nn_models import posterior_nn
import torch
from sbi import inference as inference
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math
import csv
import sbibm
import argparse
from sbibm.algorithms.sbi.snpe import wrap_posterior
from torch import nn



def build_networks_ensemble(nb_networks, neural_net= "made",hidden_features = 50,z_score_x= "independent",z_score_theta = "independent",):
    networks_ensemble = []
    for _ in range(nb_networks):
        # mdn = build_mdn(batch_x=batch_x,batch_y=batch_theta, num_components=num_components)
        # networks_ensemble.append(mdn)
        density_estimator_fun = posterior_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
        #sortie de mean layer pour mdn
        num_components = 10,
        #nb de bloc pour maf et nsf
        num_transforms = 5,
        #out features pour les splines pour nsf
        num_bins= 10,
        embedding_net=nn.Identity())
        networks_ensemble.append(density_estimator_fun)
    return networks_ensemble

def acquisition_theta(theta, x_0, density_estimators):

    mean = 0
    posteriors =torch.zeros(1)
    
    for i in range (len(density_estimators)):
        
        posterior = torch.exp(density_estimators[i].log_prob(theta, context=x_0))
        posteriors=torch.cat((posteriors, posterior), dim=0)
        mean +=posterior/len(density_estimators)
   
    var_posterior = torch.mean((posteriors[1:] - mean)**2)
    return var_posterior

def plot_acquisition(x_0, density_estimators,prior,fig):
    n_samples = 500
    thetas = prior.sample((n_samples,)).reshape(n_samples,1,2)
    y=[]
    for t in thetas:
        y.append(-acquisition_theta(t, x_0, density_estimators).item())

    df = pd.DataFrame.from_dict({"x":thetas[:,:,0].squeeze(), "y" : thetas[:,:,1].squeeze(), "data":y})
  
    ax = fig.add_subplot(projection='3d')
    plot = ax.plot_trisurf(df["x"], df["y"], df["data"], cmap=plt.cm.viridis, alpha=0.4)
    fig.colorbar(plot)
    return ax


def train_networks(networks_ensemble, task, num_simulations, num_rounds, simulation_batch_size, training_batch_size,prop_new_theta, neural_net_name,model, num_atoms = 10):

    x_0 = task.get_observation(num_observation=1).reshape(1,2)
    prior = task.get_prior_dist()
    num_simulations_per_round = math.floor(num_simulations / num_rounds)
    simulator = task.get_simulator(max_calls=num_simulations)
    max_num_epochs = 2**31 - 1
    nb_new_theta = int(num_simulations_per_round*prop_new_theta)
    simulator_new = task.get_simulator(max_calls=nb_new_theta*(num_rounds-1))
    training_kwargs = {"num_atoms":num_atoms,"discard_prior_samples":False,
                "use_combined_loss":False, "force_first_round_loss":True}
    
    inference_methods = []
    for i in range (len(networks_ensemble)):
        inference_methods.append(inference.SNPE_C(density_estimator=networks_ensemble[i],prior=prior))

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
                theta, x, proposal=proposal).train(
                training_batch_size=training_batch_size,
                retrain_from_scratch=False,
                show_train_summary=True,
                max_num_epochs=max_num_epochs,
                **training_kwargs,
            )
            density_estimators.append(density_estimator)
            torch.save(density_estimator.state_dict(),f"state_dict_network_{i+1}_round_{r+1}.pt")
            
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

                    loss = -acquisition_theta(theta_init, x_0, density_estimators)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()                
                    epoch +=1

                    trajectory=torch.cat((trajectory, theta_init.detach()), dim=0)

                with open(f'trace_{prop_new_theta}_{neural_net_name}.txt', 'a') as f:
                    f.write(f"\nConverged at epoch {epoch}/{max_epochs} with a loss {loss.item()}")
                
                print(f"Converged at epoch {epoch}/{max_epochs} with a loss {loss.item()}")   

            ax.scatter(trajectory[:,0], trajectory[:,1], loss.item(), color="blue")

            print(f"################ Neighborhood of theta MaxVar for round {r+1} ################")

            new_theta = theta_init
            dim = theta_init.size(1)
            sigma = 0.01
            noise = torch.distributions.MultivariateNormal(torch.zeros(dim), sigma*torch.eye(dim))

            new_thetas = new_theta + noise.sample((nb_new_theta-1,))
            new_thetas = torch.cat((new_theta, new_thetas), dim=0)

            for i in range (nb_new_theta):
                ax.scatter(new_thetas[i,0].item(), new_thetas[i,1].item(), -acquisition_theta(new_thetas[i].reshape(1,2), x_0, density_estimators).item(), marker="+", color="red")
            plt.savefig(f"{model}_theta_maxvar_for_round_{r+2}_{len(networks_ensemble)}_networks_{neural_net_name}.png")

            theta, x = inference.simulate_for_sbi(
                        simulator,
                        proposal,
                        num_simulations=int(num_simulations_per_round*(1-prop_new_theta)),
                        simulation_batch_size=simulation_batch_size,
                    )
            theta=torch.cat((theta, new_thetas), dim=0).detach()

            new_x = simulator_new(new_thetas)
            x = torch.cat((x, new_x), dim=0).detach()


def posterior_sampling_per_round(task, round, nb_networks, density_estimators,inference_methods, num_posterior_samples, filename):
    with open(filename, "w",newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["0", "1", "label"])
                file.close()

    x_0 = task.get_observation(num_observation=1).reshape(1,2)
    # prior = task.get_prior_dist()
    num_posterior_samples_per_network = int(num_posterior_samples/nb_networks)
    #transforms = task._get_transforms(False)["parameters"]
    mcmc_method= "slice_np_vectorized"
    mcmc_parameters= {
            "num_chains": 100,
            "thin": 10,
            "warmup_steps": 25,
            "init_strategy": "resample",
            "init_strategy_parameters": {"num_candidate_samples": 10000,}
        }

    for i in range (nb_networks):
        density_estimators[i].load_state_dict(torch.load(f"state_dict_network_{i+1}_round_{round+1}.pt"))

        posterior = inference_methods[i].build_posterior(density_estimators[i])
        posterior.set_default_x(x_0)

        #posterior = wrap_posterior(posterior, transforms)

        posterior_samples = posterior.sample((num_posterior_samples_per_network,), num_workers=10, mcmc_method=mcmc_method, mcmc_parameters=mcmc_parameters)
        posterior_samples = pd.DataFrame(posterior_samples.detach().numpy(),columns=["0", "1"])
        posterior_samples=posterior_samples.assign(label="active")
        posterior_samples.to_csv(filename, mode="a", header=False, index=False)


def main(args):

    if args.phase=="train":
        networks_ensemble = build_networks_ensemble(args.nb_networks, neural_net=args.neural_net)
        task = sbibm.get_task(args.model)
        
        train_networks(networks_ensemble, task, num_simulations=args.nb_train, num_rounds=args.nb_rounds, 
                    simulation_batch_size=args.batch_size_simulator, training_batch_size=args.batch_size_training,
                    prop_new_theta=args.prop_new_theta, neural_net_name=args.neural_net, model=args.model, num_atoms=10)
    
    if args.phase=="sample":
        
        task = sbibm.get_task(args.model)
        prior = task.get_prior_dist()
        batch_theta = prior.sample((10,))
        simu= task.get_simulator()
        batch_x = simu(batch_theta)
        networks_ensemble = build_networks_ensemble(args.nb_networks,  neural_net=args.neural_net)
        density_estimators = []
        inference_methods=[]
        for i in range (len(networks_ensemble)):
            inference_methods.append(inference.SNPE_C(density_estimator=networks_ensemble[i],prior=prior,))
            density_estimators.append(inference_methods[i].append_simulations(
                    batch_theta, batch_x).train(training_batch_size=100,max_num_epochs=5))

        for round in range(args.nb_rounds):
            filename = f"{args.model}_{args.nb_posterior}_posterior_samples_{args.nb_networks}_networks_{args.neural_net}_{round+1}_rounds_{args.nb_train}_ntrain.csv"
            posterior_sampling_per_round(task=task, round = round,nb_networks=args.nb_networks, 
                                         density_estimators=density_estimators, inference_methods=inference_methods, 
                                         num_posterior_samples=args.nb_posterior, filename=filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simulator experience'
    )
    parser.add_argument('--phase', '-p', type=str, default="train",
                        help='Choose to "train" or "sample"')
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
    #print(build_networks_ensemble(2))