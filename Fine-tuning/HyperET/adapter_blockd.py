import torch
from torch import nn
from torch.nn import functional as F
import HyperET
from typing import Optional, Tuple
from torch.cuda.amp import autocast
from clip.model import ResidualAttentionBlock
from torch import Tensor

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)

class BlockDiagonalLinear(nn.Module):
    def __init__(self, block_size, in_features, out_features, curvature: float = 0.01):
        super(BlockDiagonalLinear, self).__init__()
        self.block_size = block_size
        self.r = int(out_features / block_size)
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = curvature #nn.Parameter(torch.tensor(curvature), requires_grad=True)
        # 创建对角分块矩阵的权重
        self.weights = nn.Parameter(torch.stack([
            torch.diag(torch.ones(block_size)) for _ in range(self.r)
        ]))

    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q
    def block_diagonal(self, R):
        if len(R.shape) == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def exp_map0(self, x: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Exponential map: map points from the tangent space at the vertex
        to the hyperboloid using the exponential map of Lorentz model.
        """
        #if torch.norm(x) < eps:
        #    return torch.zeros_like(x)
        rc_xnorm = self.curvature ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
        sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2 ** 15))
        _output = torch.sinh(sinh_input) * x / rc_xnorm
        return _output

    def expmap0(self, u):
        """
        Exponential map: map points from the tangent space at the vertex
        to the hyperboloid using the exponential map of poincare ball model.
        """
        #sqrt_c = self.curvature ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
        gamma_1 = self.tanh(self.curvature ** 0.5 * u_norm) * u / (self.curvature ** 0.5 * u_norm)
        return gamma_1

    def log_map0(self, x: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Logarithmic map: map points from the hyperboloid to the tangent space
        at the vertex using the logarithmic map of Lorentz model.
        """
        #if torch.norm(x) < eps:
        #    return torch.zeros_like(x)
        rc_x_time = torch.sqrt(1 + self.curvature * torch.sum(x ** 2, dim=-1, keepdim=True))
        _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
        rc_xnorm = self.curvature ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
        _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
        return _output

    def logmap0(self, y):
        """
        Logarithmic map: map points from the hyperboloid to the tangent space
        at the vertex using the logarithmic map of poincare ball model.
        Logarithmic map for :math:`y` from :math:`0` on the manifold.
        """
        sqrt_c = self.curvature ** 0.5
        y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), 1e-5)
        return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)

    def pairwise_inner(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute pairwise Lorentzian inner product between input vectors for 4D inputs.

        Args:
            x: Tensor of shape `(B, H, W)` giving space components of a batch of vectors 
            on the hyperboloid (where B is batch size, H and W are dimensions).
            y: Tensor of shape `(B, H, W)` giving space components of another 
            batch of points on the hyperboloid.
            curv: Positive scalar denoting negative hyperboloid curvature.

        Returns:
            Tensor of shape `(B, H, H)` giving pairwise Lorentzian inner product
            between input vectors.
        """
        # Compute the time component for both x and y (last dimension is time-like)
        x_time = torch.sqrt(1 / self.curvature + torch.sum(x**2, dim=-1, keepdim=True))
        y_time = torch.sqrt(1 / self.curvature + torch.sum(y**2, dim=-1, keepdim=True))
        
        # x @ y.T equivalent for batch processing using Einstein summation
        xyl = torch.einsum('bij,bkj->bik', x, y) - torch.einsum('bij,bkj->bik', x_time, y_time)
        
        return xyl

    def tanh(self, x, clamp=15):
        return x.clamp(-clamp, clamp).tanh()
       
    def rotation_transform(self, x):

        cosh_vals = torch.cosh(self.rotation)
        sinh_vals = torch.sinh(self.rotation)
        # 将输入 x 也 reshape 为 (N, d//2, 2)
        x = x.view(x.shape[0], -1, 2)
        # 应用 Givens 旋转
        x_rot = cosh_vals.unsqueeze(-1) * x + sinh_vals.unsqueeze(-1) * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
        # 恢复形状 (N, d)
        return x_rot.view(x.shape[0], -1)

    def reflection_transform(self, x):

        cosh_vals = torch.cosh(self.reflection)
        sinh_vals = torch.sinh(self.reflection)
        # 将输入 x 也 reshape 为 (N, d//2, 2)
        x = x.view(x.shape[0], -1, 2)
        # 应用 Givens 反射
        x_ref = cosh_vals.unsqueeze(-1) * x - sinh_vals.unsqueeze(-1) * torch.cat((x[:, :, 1:], -x[:, :, 0:1]), dim=-1)
        # 恢复形状 (N, d)
        return x_ref.view(x.shape[0], -1)

    def reflection_transform_householder(self, x):
        v = self.reflection_householder / (torch.norm(self.reflection_householder) + 1e-8)  # 归一化反射向量
        I = torch.eye(self.out_features, device=x.device)  # 单位矩阵
        H = I - 2 * torch.outer(v, v)  # Householder反射矩阵

        # 应用Householder反射
        return H @ x  # 结果转置以匹配输入的形状
    def mobius_matvec(self, m, x):
        r"""
        Generalization for matrix-vector multiplication to hyperbolic space defined as
        .. math::
            M \otimes_c x = (1/\sqrt{c}) \tanh\left(
                \frac{\|Mx\|_2}{\|x\|_2}\tanh^{-1}(\sqrt{c}\|x\|_2)
            \right)\frac{Mx}{\|Mx\|_2}
        Parameters
        ----------
        m : tensor
            matrix for multiplication
        x : tensor
            point on poincare ball
        c : float|tensor
            negative ball curvature
        Returns
        -------
        tensor
            Mobius matvec result
        """
        #c = torch.as_tensor(c).type_as(x)
        return self._mobius_matvec(m, x)


    def _mobius_matvec(self, m, x):
        x_norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
        sqrt_c = self.curvature ** 0.5
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2)
        res_c = self.tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond.bool(), res_0, res_c)
        return self._project(res)

    def mobius_add(self, x, y):
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * self.curvature * xy + self.curvature * y2) * x + (1 - self.curvature * x2) * y
        denom = 1 + 2 * self.curvature * xy + self.curvature ** 2 * x2 * y2
        return num / (denom + 1e-5)
    
    def _project(self, x):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
        maxnorm = (1 - 1e-3) / (self.curvature ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def tanh(self, x, clamp=15):
        return x.clamp(-clamp, clamp).tanh()

    def forward(self, x, visual=False):
        orig_dtype = x.dtype
        output_hyperbolic = self.expmap0(x)
        fix_filt = output_hyperbolic.data
        block_diagonal_weight = self.block_diagonal(self.weights).to(orig_dtype)
        output_hyperbolic_filt_stretch = self.mobius_matvec(block_diagonal_weight, fix_filt)
        output_euclidean = self.logmap0(output_hyperbolic_filt_stretch)
        return output_euclidean

class Adapter_qv(nn.Module):
    def __init__(self, hidden_size, dim, curvature_ratio=0.01):
        super().__init__()
        self.adapter_attn_q = BlockDiagonalLinear(block_size=dim, in_features=hidden_size, out_features=hidden_size, curvature=curvature_ratio)
        self.adapter_attn_v = BlockDiagonalLinear(block_size=dim, in_features=hidden_size, out_features=hidden_size, curvature=curvature_ratio)

        self.dim = dim
    def forward(self, attn, visual=False):
        orig_dtype = attn.dtype

        fix_filt = attn.data
        q_proj_weight, k_proj_weight, v_proj_weight = fix_filt.chunk(3, dim=0)

        filt_q = self.adapter_attn_q(q_proj_weight, visual=visual) 

        filt_v = self.adapter_attn_v(v_proj_weight, visual=visual)

        filt  = torch.cat([filt_q, k_proj_weight, filt_v], dim=0)
        return filt.to(orig_dtype)

class Adapter(nn.Module):
    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
    ):
        super().__init__()
        if hidden_dim > 0:
            self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
            self.fc2 = nn.Linear(hidden_dim, in_features, bias=False)
            self.hidden_dim = hidden_dim
            nn.init.zeros_(self.fc2.weight)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, vis_weight):
        with autocast():
            if vis_weight is not None:
                x = x @ (vis_weight[0] + vis_weight[1]).permute(0, 2, 1)
                x = self.dropout(F.silu(x))
                x = x @ (vis_weight[0] + vis_weight[2])
            else:
                x = self.fc1(x)
                x = self.dropout(F.gelu(x))
                x = self.fc2(x)
        return x


def forward_llama(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],
                  vis_weight):
    if self.training and self.gradient_checkpointing:
        h = x + torch.utils.checkpoint.checkpoint(self.attention, self.attention_norm(x), start_pos, freqs_cis, mask)
        h_norm = self.ffn_norm(h)
        out = h + torch.utils.checkpoint.checkpoint(self.feed_forward, h_norm) + self.adapter_mlp(h_norm,
                                                                                                  vis_weight) * self.s
    else:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.drop_path(
            self.feed_forward((self.ffn_norm(h))) + self.adapter_mlp(self.ffn_norm(h), vis_weight) * self.s)
    return out

def mha_io_hook(module, inputs, output):
    q = inputs[0]
    print(f"[hook] input query shape: {tuple(q.shape)}")
    print(f"[hook] output shape:      {tuple(output[0].shape) if isinstance(output, tuple) else tuple(output.shape)}")



def forward_clip(self, x:torch.Tensor):
    B, N, C = x.shape
    res_x = x
    orig_dtype = x.dtype
    qkv = self.ln_1(x)          # (B, N, C)
    W = self.hyperbolic_attn(self.attn.in_proj_weight, visual=True)

    num_heads = self.n_head
    dropout_p = getattr(self.attn, "dropout", 0.0)
    bias_k = getattr(self.attn, "bias_k", None)
    bias_v = getattr(self.attn, "bias_v", None)
    out, _ = F.multi_head_attention_forward(
        query=qkv, key=qkv, value=qkv,
        embed_dim_to_check=C,
        num_heads=num_heads,
        in_proj_weight=W,
        in_proj_bias=self.attn.in_proj_bias,
        bias_k=bias_k, bias_v=bias_v,
        add_zero_attn=False,
        dropout_p=dropout_p if self.training else 0.0,
        out_proj_weight=self.attn.out_proj.weight,
        out_proj_bias=self.attn.out_proj.bias,
        training=self.training,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None, k_proj_weight=None, v_proj_weight=None,
        average_attn_weights=False
    )

    y = out.contiguous()  # (B, N, C)
    final = res_x + y
    final = final + self.mlp(self.ln_2(final)) + self.adapter_mlp(self.ln_2(final), None) * self.s
    return final



def set_Llama_Adapter(model, s=1, gradient_checkpointing=False):
    for _ in model.children():
        if type(_) == HyperET.model.TransformerBlock:
            _.adapter_mlp = Adapter(_.dim, hidden_dim=0)
            _.s = s
            _.gradient_checkpointing = gradient_checkpointing
            bound_method = forward_llama.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Llama_Adapter(_, s, gradient_checkpointing=gradient_checkpointing)


def set_Clip_Adapter(model, dim=8, s=0.1, curv=0.01):
    for _ in model.children():
        if type(_) == ResidualAttentionBlock:
            _.hyperbolic_attn = Adapter_qv(1024, dim=4, curvature_ratio=curv)
            _.adapter_mlp = Adapter(1024, hidden_dim=dim)
            _.s = s
            bound_method = forward_clip.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Clip_Adapter(_, dim, s, curv)