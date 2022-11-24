"""Create all models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import torch.autograd as autograd
import pdb

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class ActionEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_units, layers):
        super().__init__()

        self.fcin = nn.Linear(input_dim, hidden_units)

        self.block = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for i in range(layers)])

        self.mu = nn.Linear(hidden_units, out_dim)
        self.log_std = nn.Linear(hidden_units, out_dim)

    def forward(self, x):
        x = F.relu(self.fcin(x))
        for block in self.block:
            x = F.relu(block(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = torch.exp(torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX))
        density = Normal(mu, std)
        sample = density.rsample()

        return sample, density, mu, std


class ActionDecoder(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_units, layers):
        super().__init__()

        self.fcin = nn.Linear(input_dim, hidden_units)

        self.block = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for i in range(layers)])

        self.fcout = nn.Linear(hidden_units, out_dim)

    def forward(self, x):
        x = F.relu(self.fcin(x))
        for block in self.block:
            x = F.relu(block(x))

        return self.fcout(x)    

    
class EncoderTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layers):
        super().__init__()

        self.fcin = nn.Linear(input_dim, hidden_dim)

        self.stacked1 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(layers)])

        self.stacked2 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(layers)])

        self.mu = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        z = [self.fcin(x[:, i, :]) for i in range(x.shape[1])]
        z = torch.stack(z).swapaxes(0, 1)
        a, b, c = z.shape
        pos_enc = positional_encoding(a, b, c, z)
        z = z + pos_enc

        z_list = []
        for layer in self.stacked1:
            z = [F.relu(layer(z[:, i, :])) for i in range(z.shape[1])]
            z = torch.sum(torch.stack(z), dim=0)
            z_list.append(z)

        out = 0
        for z, layer in zip(z_list, self.stacked2):
            aux = F.relu(layer(z))
            out += aux

        out /= len(self.stacked2)
        mu = self.mu(out)
        log_std = self.log_std(out)

        std = torch.exp(torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX))
        density = Normal(mu, std)
        sample = density.rsample()
        return sample, density, mu, std
       
    
class DecoderTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layers, skill_length):
        super().__init__()
        
        self.fc_seq = nn.Linear(output_dim, hidden_dim)

        self.fc_z = nn.Linear(input_dim, hidden_dim)

        self.stacked1 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(layers)])

        self.stacked2 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(layers)])

        self.fcout = nn.Linear(hidden_dim, output_dim)
        self.out_dim = output_dim
        self.hidden = hidden_dim
        self.skill_length = skill_length

    def forward(self, z):
        pos_enc = positional_encoding(z.shape[0], self.skill_length+1, self.hidden, z)

        seq = torch.zeros(z.shape[0], self.out_dim, dtype=torch.double).to(z.device)
        
        seqs = []
        
        for i in range(self.skill_length+1):
            seq = self.cell(z, pos_enc[:, i, :], seq)
            seqs.append(seq)

        return torch.stack(seqs).swapaxes(0, 1)[:, 1::, :]

    def cell(self, z, pos_enc, seq):
        emb_seq = self.fc_seq(seq)
        emb_seq = emb_seq + pos_enc

        emb_z = self.fc_z(z)
        x = emb_seq + emb_z

        x_list = []
        for layer in self.stacked1:
            x = F.relu(layer(x))
            x_list.append(x)

        out = 0
        for x_out, layer in zip(x_list, self.stacked2):
            aux = F.relu(layer(x_out))
            out += aux

        out /= len(self.stacked2)

        return self.fcout(out)
                
        
def positional_encoding(a, b, c, x):
    pos = torch.arange(b, dtype=torch.double).to(x.device)
    ind = 10**torch.arange(c//2, dtype=torch.double).to(x.device)

    pos_enc = torch.einsum('i,j->ij', pos, ind)

    sin_enc = torch.sin(pos_enc).reshape(b, -1, 1)
    cos_enc = torch.cos(pos_enc).reshape(b, -1, 1)

    pos_enc = torch.cat([sin_enc, cos_enc], dim=-1).view(b, -1)

    return pos_enc.repeat(a, 1, 1)
 
           
class Skill_prior(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim):
        super().__init__()

        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, z_dim)

        self.log_std = nn.Parameter(torch.Tensor(z_dim))

        
    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        mean = self.mean_linear(x)
        std = torch.exp(torch.clamp(self.log_std, min=LOG_SIG_MIN))

        density = Normal(mean, std)
        
        return density


class Policy(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim):
        super().__init__()

        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, z_dim)

        self.log_std = nn.Parameter(torch.Tensor(z_dim))
        self.log_std.data.fill_(math.log(1))

    
    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        mean = self.mean_linear(x)
        std = torch.exp(torch.clamp(self.log_std, min=LOG_SIG_MIN))

        density = Normal(mean, std)
        skill = density.rsample()
        log_prob = density.log_prob(skill)
        log_prob = log_prob.mean(axis=1, keepdim=True)
        
        return skill, log_prob, density, mean


class Critic(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim):
        super().__init__()

        self.linear1 = nn.Linear(obs_dim + z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, vals):
        x = F.relu(self.linear1(vals))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        value = self.mean_linear(x)

        return value
    
                

# from torch.nn.utils.stateless import functional_call
# from utils import params_extraction

# model = Encoder(6, 2, 3, 2)
# #model_d = Decoder(6, 2, 10)
# x = torch.rand(20, 3, 6)-0.5


# params = params_extraction([model], ['Encoder'])
# pdb.set_trace()
# out, _ = functional_call(model, params['Encoder'], x)
# #out, _ = model(x)

# #fout = functional_call(model_d, params['Decoder'], out)

# grad = autograd.grad(torch.mean(out), params['Encoder'].values(), retain_graph=True, allow_unused=True)

# print(grad)
