# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import Any, Callable, Dict, Optional, Union


import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.posteriors.direct_posterior import DirectPosterior
import sbi.utils as utils
from sbi.neural_nets.density_estimators.base import DensityEstimator
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.sbi_types import TensorboardSummaryWriter
from sbi.utils import del_entries
from copy import deepcopy




class SNPE_D(PosteriorEstimator):
    def __init__(
        self,
        observation: Tensor,
        regularisation: float = 0,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "mdn",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""SNPE-D [1]. 

        [1] _Estimation of Non-Normalized Statistical Models by Score Matching_,
            Aapo Hyvarinen, Journal of Machine Learning Research 2005, https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf.

        This class implements SNPE-D. SNPE-D trains across multiple rounds with a
        score matching loss. This will make training converge directly to the true posterior 
        without taking care of the normalization constant.
        

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during training.
        """
        self._regularisation = regularisation
        self._x0 = observation
        self._previous_density_estimator = []
        kwargs = del_entries(locals(), entries=("self", "__class__", "regularisation", "observation"))
        super().__init__(**kwargs)

    def _loss(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
        calibration_kernel: Callable,
        force_first_round_loss: bool = False,
        
    ) -> Tensor:
        """
        Return loss.

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Indicate if the data (theta, x) of the batch 
                are sampled from the prior or from the proposal.
            proposal: Proposal distribution.

        Returns:
            Score matching loss.
        """
        
        from torch.autograd import grad
        
        with torch.enable_grad():
            
            theta.requires_grad_(True)

            def log_q(theta_batch):
                log_q = self._neural_net.log_prob(theta_batch, x)
                if self._round > 0:
                    last_proposal = torch.zeros((len(self._previous_density_estimator), theta_batch.size(0)), device = theta_batch.device)
                    last_proposal[0] = self._previous_density_estimator[0].log_prob(theta_batch)
                    for i in range(1, len(self._previous_density_estimator)):
                        last_proposal[i] = self._previous_density_estimator[i].log_prob(theta_batch, condition=self._x0)
                    log_proposal = torch.logsumexp(input=last_proposal, dim=0)
                    log_proposal = -torch.log(torch.tensor([self._round+1])) + log_proposal

                    #log_proposal = self._previous_density_estimator[-1].log_prob(theta_batch, condition=self._x0)
                    
                    log_q += log_proposal - self._prior.log_prob(theta_batch)
                    
                return torch.sum(log_q)
            
            # def log_proposal(theta_batch):
            #     return torch.sum(self._previous_density_estimator[-1].log_prob(theta_batch, condition=self._x0))
            
            # def log_prior(theta_batch):
            #     return torch.sum(self._prior.log_prob(theta_batch))
            

            grad_theta_log_q = grad(log_q(theta), theta, create_graph=True)[0]
            
            norm_grad_theta_log_q = torch.sum(grad_theta_log_q**2, dim=1)
           
           
            laplacian_theta_log_q = torch.zeros(theta.size(0), device=theta.device)
            diag_hessian_theta_norm = torch.zeros(theta.size(0), device=theta.device)
            for i in range (theta.size(1)):
                d2_log_q_d2_theta = grad(torch.sum(grad_theta_log_q, dim=0)[i], theta, create_graph=True)[0][:,i]
                laplacian_theta_log_q += d2_log_q_d2_theta
                diag_hessian_theta_norm += d2_log_q_d2_theta**2


            loss = laplacian_theta_log_q + 0.5*norm_grad_theta_log_q + self._regularisation*diag_hessian_theta_norm

            # if self._round>0:

            #     grad_theta_log_proposal = grad(log_proposal(theta), theta, create_graph=True)[0]
            #     grad_theta_log_prior = grad(log_prior(theta), theta, create_graph=True)[0]

            #     ps_q_proposal = torch.sum(grad_theta_log_q*grad_theta_log_proposal, dim=1)
            #     ps_q_prior = torch.sum(grad_theta_log_q*grad_theta_log_prior, dim=1)

            #     if self._regularisation != 0:
            #         d2_log_proposal_d2_theta = grad(torch.sum(grad_theta_log_proposal, dim=0)[i], theta, create_graph=True)[0][:,i]
            #         d2_log_prior_d2_theta = grad(torch.sum(grad_theta_log_prior, dim=0)[i], theta, create_graph=True)[0][:,i]

                
            #     loss = loss + ps_q_proposal - ps_q_prior
                

            
            #-------------LOSS WITH GRADIENTS -- GAUSSIAN LINEAR-------------------------------------------#
        
            # posterior = MultivariateNormal(x/2, torch.eye(10)/20)
            # def logp(theta):
            #     return torch.sum(posterior.log_prob(theta) + log_proposal - self._prior.log_prob(theta))
            # d_log_p_d_theta_sum = grad(logp(new_theta), new_theta, create_graph=True)
            # loss_sum = 0.5*torch.sum((d_log_q_d_theta_sum[0]-d_log_p_d_theta_sum[0])**2, dim=1)

            #---------------------------------------------------------------------------------------------#
            
        theta.requires_grad_(False)
        
        if not self._neural_net.training:
            loss = loss.detach()
        return loss
    
    
    def _log_prob_proposal_posterior(
        self, 
        theta: Tensor, 
        x: Tensor, 
        masks: Tensor, 
        proposal: Optional[Any],
    ) -> Tensor:
        pass


    
    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> DensityEstimator:
        r"""Return density estimator that approximates directly the distribution $p(\theta|x)$.

        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round. Not supported for
                SNPE-A.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """
        kwargs = utils.del_entries(
            locals(),
            entries=(
                "self",
                "__class__",
            ),
        )
        
        self._round = max(self._data_round_index)
        if self._round==0:
            self._previous_density_estimator.append(self._prior)
        density_estimator = super().train(**kwargs)
        density_estimator_copy = deepcopy(density_estimator)
        self._previous_density_estimator.append(density_estimator_copy)
        return density_estimator


