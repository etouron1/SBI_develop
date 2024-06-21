import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import likelihood_nn

from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task
from sbibm.metrics import c2st, mmd


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_rounds: int = 10,
    neural_net: str = "maf",
    hidden_features: int = 50,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 10000,
    automatic_transforms_enabled: bool = True,
    mcmc_method: str = "slice_np_vectorized",
    mcmc_parameters: Dict[str, Any] = {
        "num_chains": 100,
        "thin": 10,
        "warmup_steps": 25,
        # NOTE: resample is the init strategy used for the main paper results.
        "init_strategy": "resample",
        # NOTE: sir kwargs changed: num_candidate_samples = num_batches * batch_size
        "init_strategy_parameters": {
            "num_candidate_samples": 10000,
        },
    },
    z_score_x: str = "independent",
    z_score_theta: str = "independent",
    max_num_epochs: int = 2**31 - 1,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs (S)NLE from `sbi`

    Args:
        task: Task instance
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_rounds: Number of rounds
        neural_net: Neural network to use, one of maf / mdn / made / nsf
        hidden_features: Number of hidden features in network
        simulation_batch_size: Batch size for simulator
        training_batch_size: Batch size for training network
        automatic_transforms_enabled: Whether to enable automatic transforms
        mcmc_method: MCMC method
        mcmc_parameters: MCMC parameters
        z_score_x: Whether to z-score x
        z_score_theta: Whether to z-score theta
        max_num_epochs: Maximum number of epochs

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    
    print("SNLE-A")
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)

    if num_rounds == 1:
        log.info(f"Running NLE")
        num_simulations_per_round = num_simulations
    else:
        log.info(f"Running SNLE")
        num_simulations_per_round = math.floor(num_simulations / num_rounds)

    if simulation_batch_size > num_simulations_per_round:
        simulation_batch_size = num_simulations_per_round
        log.warn("Reduced simulation_batch_size to num_simulation_per_round")

    if training_batch_size > num_simulations_per_round:
        training_batch_size = num_simulations_per_round
        log.warn("Reduced training_batch_size to num_simulation_per_round")
    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    simulator = task.get_simulator(max_calls=num_simulations)

    transforms = task._get_transforms(automatic_transforms_enabled)[
        "parameters"
    ]
    if automatic_transforms_enabled:
        prior = wrap_prior_dist(prior, transforms)
        simulator = wrap_simulator_fn(simulator, transforms)

    density_estimator_fun = likelihood_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )
    inference_method = inference.SNLE_A(
        density_estimator=density_estimator_fun,
        prior=prior,
    )

    posteriors = []
    proposal = prior
    import csv
    with open(f"accuracy_{num_observation}_obs.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "c2st", "mmd"])
    for r in range(num_rounds):
        theta, x = inference.simulate_for_sbi(
            simulator,
            proposal,
            num_simulations=num_simulations_per_round,
            simulation_batch_size=simulation_batch_size,
        )

        density_estimator = inference_method.append_simulations(
            theta, x, from_round=r
        ).train(
            training_batch_size=training_batch_size,
            retrain_from_scratch=False,
            discard_prior_samples=False,
            show_train_summary=True,
            max_num_epochs=max_num_epochs,
        )

        (
            potential_fn,
            theta_transform,
        ) = inference.likelihood_estimator_based_potential(
            density_estimator,
            prior,
            observation,
            # NOTE: disable transform if sbibm does it. will return IdentityTransform.
            enable_transform=not automatic_transforms_enabled,
        )
        posterior = inference.MCMCPosterior(
            potential_fn=potential_fn,
            proposal=prior,  # proposal for init_strategy
            theta_transform=theta_transform,
            method=mcmc_method,
            **mcmc_parameters,
        )
        #change
        
        import pandas as pd
        posterior_sampling = posterior.set_default_x(observation)
        posterior_sampling = wrap_posterior(posterior, transforms)
        posterior_samples = posterior_sampling.sample((num_samples,))
        # posterior_samples = pd.DataFrame(posterior_samples.detach().numpy(),columns=["0", "1"])
        # posterior_samples=posterior_samples.assign(label="active")
        # posterior_samples.to_csv(f"_{num_samples}_posterior_samples_snle-A-{neural_net}_{r+1}_round_{num_simulations}_ntrain.csv", header=False, index=False)
        reference_samples = task.get_reference_posterior_samples(num_observation=num_observation)
        accuracy_c2st = c2st(reference_samples, posterior_samples).item()
        accuracy_mmd = mmd(reference_samples, posterior_samples).item()
        with open(f"accuracy_{num_observation}_obs.csv",  "a") as f:
            writer = csv.writer(f)
            writer.writerow([r+1, accuracy_c2st, accuracy_mmd])
        #change

        # Change init_strategy to latest_sample after second round.
        if r > 1:
            posterior.init_strategy = "latest_sample"
            # copy init params from round 2 posterior.
            posterior._mcmc_init_params = posteriors[-1]._mcmc_init_params

        proposal = posterior.set_default_x(observation)
        posteriors.append(posterior)

    posterior = wrap_posterior(posteriors[-1], transforms)

    assert simulator.num_simulations == num_simulations

    samples = posterior.sample((num_samples,)).detach()

    return samples, simulator.num_simulations, None