import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CatELU(nn.Module):
    """
    CatELU activation (feature-doubling version).

    Applies ELU to the input and its negation,
    doubling the feature dimension.

    Output: [..., 2 * D] given input [..., D]

    NOTE:
        This is a structural activation and NOT element-wise.
        Assumes the last dimension is the feature dimension.
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.elu = nn.ELU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() >= 2, \
            f"CatELU expects feature dimension in the last axis, got shape {x.shape}"

        y1 = self.elu(x)
        y2 = self.elu(-x)
        return torch.cat((y1, y2), dim=-1)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "cat_elu":
        return CatELU()
    else:
        print("invalid activation function!")
        return None

class L2Norm(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=-1)

class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self):
        super().__init__()
        self.dim = 8  # for latent dim 512

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"

# MLP implementation for MoE
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation='elu', last_activation: str | None = None):
        super().__init__()

        dims = [input_dim] + hidden_dims
        act_func = get_activation(activation)
        layers = []
        last_dim = dims[0]
        for h_dim in dims[1:]:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(act_func)
            if activation == 'cat_elu':
                last_dim = h_dim * 2
            else:
                last_dim = h_dim
            
        if isinstance(output_dim, int):
            layers.append(nn.Linear(last_dim, output_dim))
        elif isinstance(output_dim, tuple) or isinstance(output_dim, list):
            layers.append(nn.Linear(last_dim, np.prod(output_dim)))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))
        else:
            raise ValueError("output_dim must be int, tuple or list")        
        
        if last_activation is not None:
            last_act_func = get_activation(last_activation)
            layers.append(last_act_func)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Experts(nn.Module):
    def __init__(self,
                 expert_num,
                 input_dim,
                 backbone_hidden_dims,
                 expert_hidden_dim,
                 output_dim,
                 activation='elu',
    ):
        super().__init__()
        self.expert_num = expert_num
        self.output_dim = output_dim

        self.backbone = MLP(input_dim, expert_num * expert_hidden_dim, backbone_hidden_dims, activation, last_activation=activation)
        self.experts = nn.Conv1d(
            in_channels=expert_num*expert_hidden_dim if activation != 'cat_elu' else expert_num*expert_hidden_dim*2,
            out_channels=expert_num*output_dim,
            kernel_size=1,
            groups=expert_num,
        )
    
    def forward(self, x):
        shared_features = self.backbone(x).unsqueeze(-1)  # (B, expert_num * expert_hidden_dim, 1)
        expert_outs = self.experts(shared_features).squeeze(-1)  # (B, expert_num * output_dim)
        expert_outs = expert_outs.reshape(-1, self.expert_num, self.output_dim)
        return expert_outs

class MoE(nn.Module):
    def __init__(self,
                 expert_num,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 activation='elu',
    ):
        super().__init__()

        # Expert networks
        self.experts = Experts(
            expert_num=expert_num,
            input_dim=input_dim,
            backbone_hidden_dims=hidden_dims[:-1],
            expert_hidden_dim=hidden_dims[-1],
            output_dim=output_dim,
            activation=activation,
        )
        
        # Gating network
        self.gating_network = nn.Sequential(
            MLP(input_dim, expert_num, hidden_dims, activation),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        weights = self.gating_network(x)  # (B, expert_num)
        expert_outs = self.experts(x)  # (B, expert_num, output_dim)
        output = torch.sum(weights.unsqueeze(-1) * expert_outs, dim=1)  # (B, output_dim)
        return output, weights

