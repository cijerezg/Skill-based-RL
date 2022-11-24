import torch.autograd as autograd
import torch
import copy
import torch.nn as nn
import gym
from collections import OrderedDict
from torch.nn.utils.stateless import functional_call
import numpy as np
import seaborn as sns
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import utils
import pdb
import wandb


class hyper_params:
    def __init__(self, args):
        # Batch sizes
        self.meta_batch_size = args.meta_batch_size
        self.batch_size = args.batch_size
        self.vae_batch_size = args.vae_batch_size

        # VAE hierarchy. It should be a dict.
        self.hrchy = self.creating_hierarchy(args)
        self.skill_length = np.prod([lev['length'] for lev in self.hrchy.values()])
        self.z_action = args.z_action
        
        # Learning rates
        self.rl_lr = args.rl_lr
        self.vae_lr = args.vae_lr

        # RL hyperparameters:
        self.discount = args.discount
        self.alpha = args.alpha

        # Env hyperparameters
        self.env_id = args.env_id
        self.action_dim, self.state_dim = self.env_dims(args.env_id)

        # General
        self.device = torch.device(args.device)
        self.hidden_dim = args.hidden_dim  # Hidden dimension for all NNs

    def env_dims(self, env_id):
        env = gym.make(env_id)
        action_dim = env.action_space.shape[0]
        state_dim = env.observation_space.shape[0]
        env.close()
        del env
        return action_dim, state_dim

    def creating_hierarchy(self, args):
        hrchy = {}
        for i in range(args.levels):
            hrchy[i] = {'length': args.level_length, 'z': args.z_vae}

        return hrchy

    
def gradient(loss: torch.tensor,
             params: list,
             name,
             second_order: bool = False,
             ) -> torch.tensor:
    """Compute gradient.

    Compute gradient of loss with respect to parameters.

    Parameters
    ----------
    loss : torch.tensor
        Scalar that depends on params.
    params : list
        Sequence of tensors that the gradient will be computed with
        respect to
    second_order : bool
        Select to compute second or higher order derivatives.

    Returns
    -------
    torch.tensor
        Flattened gradient.

    Examples
    --------
    loss = torch.abs(model(y) - y)
    grad = gradient(loss, model.parameters())

    """
    grad = autograd.grad(loss, params.values(), retain_graph=True,
                         create_graph=second_order, allow_unused=True)

    # if 'Prior' not in name and 'Action' not in name and 'Seq' not in name:
    #     wandb.log({f' Grad {name}':
    #                wandb.Histogram(nn.utils.parameters_to_vector(grad).cpu())})

    if 'Critic' in name or 'Policy' in name:
        wandb.log({f' Grad norm {name}': torch.norm(nn.utils.parameters_to_vector(grad)).cpu()})
        # wandb.log({f' Grad {name}':
        #            wandb.Histogram(nn.utils.parameters_to_vector(grad).cpu())})

    return nn.utils.parameters_to_vector(grad)


def params_update_deep(params: list,
                       grad: torch.tensor,
                       lr: float
                       ) -> list:
    """Apply gradient descent update to params.

    It creates a deepcopy of params to save the updated params. This
    is useful for higher order derivatives.

    Parameters
    ----------
    params : list
        Sequences of tensors. These are the base parameters that will
        be updated.
    grad : torch.tensor
        Flattened tensor containing the gradient.
    lr : float
        Learning rate.

    Returns
    -------
    list
        The updated parameters.

    Examples
    --------
    grad = torch.ones(10)
    w, b = torch.rand(5), torch.rand(b)
    params = {'weight': nn.Parameter(w), 'bias': nn.Parameter(b)}
    lr = 0.1
    new_params = params_update_deep(params, grad, lr)

    """
    params_updt = copy.deepcopy(params)
    start, end = 0, 0
    for name, param in params.items():
        start = end
        end = start + param.numel()
        update = grad[start:end].reshape(param.shape)
        params_updt[name] = param - lr * update
    return params_updt

    
def params_update_shalllow(params: list,
                           grad: torch.tensor,
                           lr: float
                           ) -> list:
    """Apply gradient descent update to params.

    It creates rewrites the params to save the updated params. Do not use this
    when computing higher order derivatives.
    is useful for higher order derivatives.

    Parameters
    ----------
    params : list
        Sequences of tensors. These are the base parameters that will
        be updated.
    grad : torch.tensor
        Flattened tensor containing the gradient.
    lr : float
        Learning rate.

    Returns
    -------
    list
        The updated parameters.

    Examples
    --------
    grad = torch.ones(10)
    w, b = torch.rand(5), torch.rand(b)
    params = {'weight': nn.Parameter(w), 'bias': nn.Parameter(b)}
    lr = 0.1
    new_params = params_update_deep(params, grad, lr)

    """
    params_updt = copy.copy(params)
    start, end = 0, 0
    for name, param in params.items():
        start = end
        end = start + param.numel()
        update = grad[start:end].reshape(param.shape)
        params_updt[name] = param - lr * update
    return params_updt


def GD_full_update(params: dict,
                   losses: list,
                   keys: list,
                   lr: float,
                   ) -> list:
    """Compute GD for multiple models.

    Compute and update parameters of models in the params list with
    respect to the losses in the loss list. Do not use for second or
    higher order derivatives.

    Parameters
    ----------
    params : list
        Each element of the dictionary has the parameters of a model.
    losses : list
        Each element of the list is a scalar tensor. It should match
        the order of the params dict, e.g., first element of loss,
        should correspond to first params.
    lr : float
        Learning rate.

    Returns
    -------
    list
        Update dictionary with all parameters.

    """
    for loss, key in zip(losses, keys):
        grad = gradient(loss, params[key], key)
        # print(f'Norm of {key} is {torch.norm(grad)}')
        if key == 'Encoder':
            lr = lr #* 10
        params[key] = params_update_shalllow(params[key], grad, lr)
    return params


def params_extraction(models: list,
                      names: list,
                      pretrained_params,
                      ) -> dict:
    """Get and init params from model to use with functional call.

    The models list contains the pytorch model. The parameters are
    initialized with bias and std 0, and rest with orthogonal init.

    Parameters
    ----------
    models : list
        Each element contains the pytorch model.
    names : list
        Strings that contains the name that will be assigned.

    Returns
    -------
    dict
        Each dictionary contains params ready to use with functional
        call.

    Examples
    --------
    See vae.py for an example.

    """
    params = OrderedDict()
    for model, name_m, pre_params in zip(models, names, pretrained_params):
        par = {}
        if pre_params is None:
            for name, param in model.named_parameters():
                if 'bias' in name:
                    init = torch.nn.init.constant_(param, 0.0)
                elif 'std' in name:
                    init = torch.nn.init.constant_(param, 0.0)
                else:
                    init = torch.nn.init.xavier_normal_(param, gain=1.0)
                par[name] = nn.Parameter(init)
        else:
            for name, param in model.named_parameters():
                init = pre_params[name]
                par[name] = nn.Parameter(init)
        params[name_m] = par
                
    return params


def load_pretrained_models(c):
    pretrained_params = []
    use_pretrained = False

    if c.use_pretrained_action_VAE:
        params_act_encoder = torch.load('trained_models/ActionEncoder_MyData_z2.pt')
        params_act_decoder = torch.load('trained_models/ActionDecoder_MyData_z2.pt')
        pretrained_params.append(params_act_encoder)
        pretrained_params.append(params_act_decoder)
        use_pretrained = True
            
    if c.use_pretrained_VAE:
        vae_mod = torch.load(f'VAE_models/l{c.levels}_len{c.level_length}_z{c.z_vae}/params.pt')
        pretrained_params.extend([*vae_mod.values()])
        use_pretrained = True

    if not use_pretrained:
        pretrained_params.append(None)

    return pretrained_params

def vae_config(config):
    if config.case == 0:
        config.levels = 3
        config.level_length = 4
        config.vae_lr = 0.01
        config.z_vae = 16

    elif config.case == 1:
        config.levels = 2
        config.level_length = 8
        config.vae_lr = 0.01
        config.z_vae = 16

    elif config.case == 2:
        config.levels = 1
        config.level_length = 64
        config.vae_lr = 0.01
        config.z_vae = 16
        
    return config
