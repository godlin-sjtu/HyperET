import torch
import torch.nn as nn
import math
from granularity_vis import hyperbolic_norm

def expmap0(u: torch.Tensor, c: float = 1.0, epsilon: float = 1e-6) -> torch.Tensor:
    sqrt_c = math.sqrt(c)
    u_norm = u.norm(dim=-1, keepdim=True)
    u_norm_clamped = torch.clamp_min(u_norm, epsilon)
    theta = sqrt_c * u_norm_clamped
    scale = torch.tanh(theta) / theta
    return scale * u

def hyperbolic_norm(img_feats: torch.Tensor, curv: float = 0.01) -> tuple:
    sqrt_c = math.sqrt(curv)
    x = expmap0(img_feats, c=curv)

    #the pre-tanh version of the hyperbolic radius, which is easier to visualize.
    x_norm = torch.linalg.norm(x, dim=-1)
    dist_mean = x_norm.mean()
    return x_norm, dist_mean