# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack
from functools import partial
from collections import namedtuple
from torch import Tensor, nn, einsum
import numpy as np
from torch.cuda.amp import autocast

__all__ = [
    "window_partition",
    "window_unpartition",
    "add_decomposed_rel_pos",
    "get_abs_pos",
    "PatchEmbed",
    "CoordinateDescentRouter",
    "LinearRouting",
    "DynamicRouter",
    "coor_descent",
]

RouterReturn = namedtuple('RouterReturn', ['indices', 'scores', 'routed_tokens', 'routed_mask'])

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seq_len = tensor.shape[dim]
    m = seq_len / multiple
    if m.is_integer():
        return tensor, seq_len

    remainder = math.ceil(m) * multiple - seq_len
    pad_offset = (0,) * (-1 - dim) * 2
    padded_tensor = F.pad(tensor, (*pad_offset, 0, remainder), value = value)
    return padded_tensor, seq_len

def batched_gather(x, indices):
    batch_range = create_batch_range(indices, indices.ndim - 1)
    return x[batch_range, indices]

def identity(t):
    return t

def l2norm(t):
    return F.normalize(t, dim = -1)

# tensor helpers

def create_batch_range(t, right_pad_dims = 1):
    b, device = t.shape[0], t.device
    batch_range = torch.arange(b, device = device)
    pad_dims = ((1,) * right_pad_dims)
    return batch_range.reshape(-1, *pad_dims)


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size # 80, 120
    k_h, k_w = k_size # 80, 120
    Rh = get_rel_pos(q_h, k_h, rel_pos_h) # torch.Size([80, 80, 64])
    Rw = get_rel_pos(q_w, k_w, rel_pos_w) # torch.Size([120, 120, 64])

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh) # torch.Size([6, 80, 120, 80])
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw) # torch.Size([6, 80, 120, 120])

    attn = ( # 2048*2048这里会爆, rel_h: [6, 128, 128, 128], rel_w:[6, 128, 128, 128], None扩充后都乘128：6*128*128*128*128*4 = 6,442,450,944
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

class CoordinateDescentRouter(nn.Module):
    """
    from Wright et al. https://arxiv.org/abs/1502.04759
    then adopted by https://arxiv.org/abs/2211.01267 for multi-vector document retrieval by Qian et al
    finally, used successfully by this paper for routing to heavy branch attention / feedforward
    """

    def __init__(
        self,
        dim,
        straight_through = True,
        n_iters = 20,                   # 20 iterations in a new paper, utilizing ε-scaling
        fetch_k_ratio = 9 / 8,          # in the paper, they do a bit slightly higher k (times this ratio) for better learning
        eps = 0.03,                     # the epsilon for coordinate descent. in a recent paper, they used 0.03 for text and 1.0 for speech
        eps_decay = 0.7,
        eps_init = 4.,
        num_routing_tokens = 1,
        learned_routing_tokens = False,
        use_triton = False,
        cosine_sim_routing = False,
        cosine_sim_scale = 8,
        route_block_size = None,
        triton_checkpoint_segments = None # whether to recompute the coordinate descent in segments, with 4 and 50 iterations, backwards is sped up 3x times at the expense of forwards and some memory for saving initial a and b
    ):
        super().__init__()
        assert fetch_k_ratio >= 1.

        self.n_iters = n_iters
        self.fetch_k_ratio = fetch_k_ratio

        self.coor_descent = coor_descent

        # epsilon related hparams, for ε-scaling

        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_init = eps_init

        if use_triton:
            from triton_coor_descent import triton_coor_descent
            triton_checkpoint_segments = default(triton_checkpoint_segments, n_iters // 5)
            self.coor_descent = partial(triton_coor_descent, checkpoint_segments = triton_checkpoint_segments)

        self.is_one_routing_token = num_routing_tokens == 1
        self.num_routing_tokens = num_routing_tokens

        self.route_block_size = route_block_size

        self.routing_token = nn.Parameter(torch.randn(num_routing_tokens, dim)) if not learned_routing_tokens else None
        self.straight_through = straight_through

        # whether to use cosine sim for routing

        self.cosine_sim_routing = cosine_sim_routing
        self.cosine_sim_scale = cosine_sim_scale

    def route_back(self, src, routed_tokens, indices):
        batch_range = create_batch_range(routed_tokens)
        src[batch_range, indices] = routed_tokens
        return src

    def forward(
        self,
        x,
        *,
        num_tokens,
        mask = None,
        random_route = False,
        routing_tokens = None,
        keep_one_route_dim = False  # if only one route, whether to keepdim
    ):
        n, device, eps, eps_init, eps_decay, num_routes, route_block_size = x.shape[-2], x.device, self.eps, self.eps_init, self.eps_decay, self.num_routing_tokens, self.route_block_size

        # do not route if the sequence length is less than the number of tokens

        has_route_dim = keep_one_route_dim or not self.is_one_routing_token

        if n <= num_tokens:
            b = x.shape[0]
            r = self.num_routing_tokens

            if has_route_dim:
                scores_shape = (b, r, n)

                x = repeat(x, 'b n d -> b r n d', r = r)

                if exists(mask):
                    mask = repeat(mask, 'b n -> b r n', r = r)
            else:
                scores_shape = (b, n)

            scores = torch.ones(scores_shape, device = device, dtype = x.dtype)

            return RouterReturn(None, scores, x, mask)

        # whether to route even amounts from blocks of the sequence

        if exists(route_block_size):
            num_blocks = n // route_block_size
            prev_seq_mult = num_blocks * route_block_size

            # just curtail to last multiple of route block size

            x = x[:, :prev_seq_mult]

            # group sequence into blocks to route

            x = rearrange(x, 'b (n w) d -> (b n) w d', w = route_block_size)

            if exists(mask):
                mask = mask[:, :prev_seq_mult]
                mask = rearrange(mask, 'b (n w) -> (b n) w', w = route_block_size)

            n = route_block_size
            num_tokens = math.ceil(num_tokens / num_blocks)

        # s stands for eventual normalized score

        maybe_l2norm = l2norm if self.cosine_sim_routing else identity

        if exists(self.routing_token):
            s = einsum('b n d, r d -> b r n', maybe_l2norm(x), maybe_l2norm(self.routing_token))
        else:
            assert exists(routing_tokens)

            if routing_tokens.ndim == 2:
                routing_tokens = rearrange(routing_tokens, 'b d -> b 1 d')

            s = einsum('b n d, b r d -> b r n', maybe_l2norm(x), maybe_l2norm(routing_tokens))

        if self.cosine_sim_routing:
            s = s * self.cosine_sim_scale

        # merge routing dimension into batch

        x = repeat(x, 'b ... -> (b r) ...', r = num_routes)
        s, ps = pack_one(s, '* n') # 进入前s.shape = 2,1,32768

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b r) ...', r = num_routes)

        # k, which controls the sparsity of the outputted scores from iterative coordinate descent

        effective_k = min(num_tokens * self.fetch_k_ratio, n)

        # coordinate descent

        scores = self.coor_descent(
            s,
            n_iters = self.n_iters,
            mask = mask,
            k = effective_k,
            eps = eps,
            eps_init = eps_init,
            eps_decay = eps_decay
        )

        # force random routing, if negative control

        if random_route:
            scores = torch.randn_like(scores)
            scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)

        # get the topk scores and indices from the sparse matrix

        selected_scores, selected_indices = scores.topk(num_tokens, dim = -1)

        if self.straight_through:
            # this would make sure all normalized scores returned are 1., but still differentiable using straight-through trick
            selected_scores = selected_scores + (1. - selected_scores).detach()

            if exists(mask):                
                selected_mask = batched_gather(mask, selected_indices)
                selected_scores = selected_scores.masked_fill(~selected_mask, 0.)

        # split out routing dimension again if need be

        if has_route_dim:
            selected_scores = unpack_one(selected_scores, ps, '* n')
            selected_indices = unpack_one(selected_indices, ps, '* n')

        # undo the windowing, if one were routing uniformly in blocks

        if exists(route_block_size):
            selected_scores = rearrange(selected_scores, '(b n) ... w -> b ... (n w)', n = num_blocks)
            selected_indices = rearrange(selected_indices, '(b n) ... w -> b ... n w', n = num_blocks)

            indices_offset = torch.arange(num_blocks, device = device) * route_block_size
            selected_indices = selected_indices + rearrange(indices_offset, 'n -> n 1')
            selected_indices = rearrange(selected_indices, 'b ... n w -> b ... (n w)')

        # auto-gather the routed tokens and mask (if not None)

        routed_tokens = batched_gather(x, selected_indices)

        routed_mask = None
        if exists(mask):
            routed_mask = batched_gather(mask, selected_indices)

        # return indices, scores, routed tokens and mask

        return RouterReturn(selected_indices, selected_scores, routed_tokens, routed_mask)

class LinearRouting(nn.Module):
    def __init__(
        self,
        in_channels,
        num_experts = 2,
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gating_network = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, num_experts)
        )
        # self.fc = nn.Linear(in_channels, in_channels / 2)

        # self.gumbel_softmax = torch.nn.functional.gumbel_softmax()

    def forward(self, x, prev_msa):
        x = torch.cat((x, prev_msa), dim = -1).permute(0, 3, 1, 2)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)

        x = self.gating_network(x)
        x = torch.nn.functional.gumbel_softmax(x, hard=True, dim=-1)
        return x
    # torch.nn.functional.softmax(x,dim=-1)

class DynamicRouter(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy=None):
        x = self.in_conv(x) # 16, 1024, 384
        B, N, C = x.size()
        local_x = x[:,:, :C//2] # 16, 1024, 192
        global_x = (x[:,:, C//2:]).sum(dim=1, keepdim=True) / N # 16, 1, 192
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1) # 16, 1024, 384
        return self.out_conv(x) # 16, 1024, 2
    
class DynamicRouter_wo_local_global(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy=None):
        x = self.in_conv(x) # 16, 1024, 384
        B, N, C = x.size()
        # local_x = x[:,:, :C//2] # 16, 1024, 192
        # global_x = (x[:,:, C//2:]).sum(dim=1, keepdim=True) / N # 16, 1, 192
        # x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1) # 16, 1024, 384
        return self.out_conv(x) # 16, 1024, 2


class SmallRouter(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2),
            nn.LogSoftmax(dim=-1)
        )
    
    def forward(self, x, policy=None):
        return self.out_conv(x)

class TinyRouter(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy=None):
        return self.out_conv(x)

class SERouter(nn.Module):
    def __init__(self, embed_dim=384, r = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim//r, embed_dim, bias=False),
            nn.Sigmoid(),
        )
        self.out_fc = nn.Sequential(
            nn.Linear(embed_dim, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy=None):
        b, c , _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)
        y = self.out_fc(y.reshape(b, c, -1).permute(0, 2, 1))
        return y


class RandomRouter(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()

    def forward(self, x, policy=None):

        shape = (x.shape[0], x.shape[1], 1)
        # 计算所需1的数量
        num_elements = torch.prod(torch.tensor(shape)).item()
        num_ones = int(num_elements * 0.25)

        # 创建一个包含0和1的列表
        score = [1] * num_ones + [0] * (num_elements - num_ones)

        # 打乱列表顺序
        # torch.manual_seed(42)  # 设置随机种子以确保可重复性
        random_score = torch.tensor(score).float().to(x.device)
        random_score = random_score[torch.randperm(num_elements)]

        # 重塑张量
        random_score = random_score.view(*shape)
        random_score = torch.cat((random_score, 1-random_score), dim = -1)

        return random_score


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

@autocast(enabled = False)
def coor_descent(
    s,
    *,
    n_iters,
    k,
    eps = 1e-1,
    eps_init = None,
    eps_decay = 1.,
    mask = None
):
    """
    coordinate descent  - https://arxiv.org/abs/1502.04759, utilized in https://arxiv.org/abs/2303.09752
    ε-scaling           - https://arxiv.org/abs/1610.06519, utilized in https://arxiv.org/abs/2304.04947

    in a follow up paper applying coordinate descent routing to efficient fine tuning
    they were able to cut n_iters from 50 -> 20 by setting eps_init = 4 and eps_decay = 0.7
    eps was dependent on the task, and ranged from 0.02 to 1
    """

    assert n_iters > 0

    mask_value = -torch.finfo(s.dtype).max

    if not isinstance(k, torch.Tensor):
        k = torch.Tensor([k]).to(s)
    else:
        k = rearrange(k, '... -> ... 1')

    logk = log(k)

    if exists(mask):
        s = s.masked_fill(~mask, mask_value) #将mask中为True位置的元素替换成给定的value

    a = 0
    b = -s

    current_eps = max(default(eps_init, eps), eps)

    for _ in range(n_iters):
        sb = ((s + b) / current_eps)

        if exists(mask):
            sb = sb.masked_fill(~mask, mask_value)

        a = current_eps * (logk - sb.logsumexp(dim = -1, keepdim = True))
        b = -F.relu(s + a)

        current_eps = max(current_eps * eps_decay, eps)

    scores = ((s + a + b) / current_eps).exp()

    if exists(mask):
        scores = scores.masked_fill(~mask, 0.)

    return scores
