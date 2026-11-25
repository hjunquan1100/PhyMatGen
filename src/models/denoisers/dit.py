"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Adapted from: https://github.com/facebookresearch/DiT
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_dim, frequency_embedding_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_dim, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, 0, labels)
        # NOTE: 0 is the label for the null class
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        # Clamp labels to valid range [0, num_classes + use_cfg_embedding - 1]
        max_idx = self.num_classes + (1 if use_dropout else 0) - 1
        labels = torch.clamp(labels, min=0, max=max_idx)
        embeddings = self.embedding_table(labels)
        return embeddings


def get_pos_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine poDiTional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., num_tokens] of type integer
        emb_dim: embedding dimension
        max_len: maximum length

    Returns:
        poDiTional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


#################################################################################
#                               Transformer blocks                              #
#################################################################################


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


def modulate(x, shift, scale):
    # TODO this is global modulation; explore per-token modulation
    d = x.shape[-1]
    B = x.shape[0]
    if d == 0:
        return x
    # 统一为 2D (batch, dim)，避免 3D conditioning 导致后续切片错误
    if shift.dim() > 2:
        shift = shift.reshape(-1, shift.shape[-1])
    if scale.dim() > 2:
        scale = scale.reshape(-1, scale.shape[-1])
    # 截断到 d 维并保证形状 (B, d)
    def _to_bd(t):
        if t.numel() == 0:
            return torch.zeros(B, d, device=x.device, dtype=x.dtype)
        if t.dim() == 1:
            t = t.unsqueeze(-1).expand(-1, d)
        else:
            t = t[..., :d] if t.shape[-1] >= d else F.pad(t, (0, d - t.shape[-1]))
        if t.shape[0] != B:
            if t.shape[0] == 0:
                return torch.zeros(B, d, device=x.device, dtype=x.dtype)
            need = (B + t.shape[0] - 1) // t.shape[0]
            t = t.repeat(need, 1)[:B].contiguous()
        return t.contiguous()
    shift = _to_bd(shift)
    scale = _to_bd(scale)
    # 最终保证 (B, d)
    if shift.shape != (B, d):
        out_shift = torch.zeros(B, d, device=x.device, dtype=x.dtype)
        r0, r1 = min(B, shift.shape[0]), min(d, shift.shape[-1])
        if r0 > 0 and r1 > 0:
            src = shift[:r0, :r1].reshape(r0, r1).to(device=x.device, dtype=x.dtype)
            out_shift[:r0, :r1] = src
        shift = out_shift
    if scale.shape != (B, d):
        out_scale = torch.zeros(B, d, device=x.device, dtype=x.dtype)
        r0, r1 = min(B, scale.shape[0]), min(d, scale.shape[-1])
        if r0 > 0 and r1 > 0:
            src = scale[:r0, :r1].reshape(r0, r1).to(device=x.device, dtype=x.dtype)
            out_scale[:r0, :r1] = src
        scale = out_scale
    shift = shift.reshape(B, d)
    scale = scale.reshape(B, d)
    out = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    # 强制与 x 同形 (B, N, d)，避免 conditioning batch 与 x 不一致导致 4D 或 wrong size
    if out.shape != x.shape:
        out = out[: B, : x.shape[1], : d].contiguous()
    return out


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=0, bias=True, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c, mask):
        # c (B, d_model), x (B, N, d_model)；强制 3D 与 d_model 一致，避免 9206 vs 768 等
        if x.dim() > 3:
            x = x.reshape(x.size(0), x.size(1), -1)
        d = x.shape[-1]
        d_model = self.attn.embed_dim
        if d != d_model:
            x = x[..., : d_model].contiguous()
            d = d_model
        c = c.reshape(-1, c.shape[-1])[: x.size(0), : d_model].contiguous()
        mod = self.adaLN_modulation(c)
        need = 6 * d
        if mod.shape[1] < need:
            mod = F.pad(mod, (0, need - mod.shape[1]))
        shift_msa = mod[:, 0 * d : 1 * d]
        scale_msa = mod[:, 1 * d : 2 * d]
        gate_msa = mod[:, 2 * d : 3 * d]
        shift_mlp = mod[:, 3 * d : 4 * d]
        scale_mlp = mod[:, 4 * d : 5 * d]
        gate_mlp = mod[:, 5 * d : 6 * d]
        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        key_mask = mask if mask.dim() == 2 else mask.reshape(mask.shape[0], -1)
        key_mask = key_mask[:, : x.shape[1]]
        attn_out = self.attn(_x, _x, _x, key_padding_mask=key_mask, need_weights=False)[0]
        if attn_out.shape[-1] != d:
            attn_out = attn_out[..., : d].contiguous()
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        if mlp_out.shape[-1] != d:
            mlp_out = mlp_out[..., : d].contiguous()
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """Diffusion model with a Transformer backbone.

    Args:
        d_x (int): Input dimension
        d_model (int): Model dimension
        num_layers (int): Number of Transformer layers
        nhead (int): Number of attention heads
        mlp_ratio (float): Ratio of hidden to input dimension in MLP
        class_dropout_prob (float): Probability of dropping class labels for classifier-free guidance
        num_datasets (int): Number of datasets for classifier-free guidance
        num_spacegroups (int): Number of spacegroups for classifier-free guidance
    """

    def __init__(
        self,
        d_x=8,
        d_model=384,
        num_layers=12,
        nhead=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_datasets=2,  # clf-free guidance input
        num_spacegroups=230,  # clf-free guidance input
    ):
        super().__init__()
        self.d_x = d_x
        self.d_model = d_model
        self.nhead = nhead

        self.x_embedder = nn.Linear(2 * d_x, d_model, bias=True)
        self.t_embedder = TimestepEmbedder(d_model)
        self.dataset_embedder = LabelEmbedder(num_datasets, d_model, class_dropout_prob)
        self.spacegroup_embedder = LabelEmbedder(num_spacegroups, d_model, class_dropout_prob)

        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, nhead, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(d_model, d_x)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.dataset_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.spacegroup_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, dataset_idx, spacegroup, mask, x_sc=None):
        """Forward pass of DiT.

        Args:
            x (torch.Tensor): Input data tensor (B, N, d_in)
            t (torch.Tensor): Time step for each sample (B,)
            dataset_idx (torch.Tensor): Dataset index for each sample (B,)
            spacegroup (torch.Tensor): Spacegroup index for each sample (B,)
            mask (torch.Tensor): True if valid token, False if padding (B, N)
            x_sc (torch.Tensor): Self-conditioning x (B, N, d_in)
        """
        # 强制 x 为 (B, N, d_in)，避免 4D 导致后续 MHA/dim3 报错
        if x.dim() > 3:
            x = x.reshape(x.size(0), x.size(1), -1)
        if x.shape[-1] != self.d_x:
            x = x[..., : self.d_x].contiguous()
        B, N = x.shape[0], x.shape[1]
        # 保证 mask 为 (B, N)，否则 pos_emb 会变成 4D 或与 x 的 (B,N) 不一致导致加法和 MHA 报错
        if mask.dim() > 2:
            mask = mask.reshape(mask.size(0), -1)
        mask = mask[:B, :N].contiguous()
        if mask.size(0) < B or mask.size(1) < N:
            new_mask = torch.zeros(B, N, dtype=mask.dtype, device=mask.device)
            new_mask[: mask.size(0), : mask.size(1)] = mask
            mask = new_mask
        # Positonal embedding，与 x 的 (B, N) 一致
        token_index = torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1
        pos_emb = get_pos_embedding(token_index, self.d_model)
        if pos_emb.dim() > 3:
            pos_emb = pos_emb.reshape(pos_emb.size(0), pos_emb.size(1), -1)

        # Self-conditioning and input embeddings: (B, N, d)
        if x_sc is None:
            x_sc = torch.zeros_like(x)
        x_emb = self.x_embedder(torch.cat([x, x_sc], dim=-1))
        # 强制 pos_emb 与 x_emb 同形，避免 data/conditioning batch 不一致导致 21 vs 210 等
        if pos_emb.shape != x_emb.shape:
            pos_emb = pos_emb[: x_emb.size(0), : x_emb.size(1), : x_emb.size(2)].contiguous()
        x = x_emb + pos_emb

        # Conditioning embeddings，强制与 x 的 batch 维一致，避免 DDP/数据导致 (9206,) 等异常
        t = self.t_embedder(t.squeeze(1))  # (B, d)
        d = self.dataset_embedder(dataset_idx, self.training)  # (B, d)
        s = self.spacegroup_embedder(spacegroup, self.training)  # (B, d)
        c = (t + d + s).reshape(-1, self.d_model)[:B, :].contiguous()  # 强制 (B, d_model)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, ~mask)  # (B, N, d)

        # Prediction layer
        x = self.final_layer(x, c)  # (B, N, d_out)
        if x.shape[-1] != self.d_x:
            x = x[..., : self.d_x].contiguous()
        x = x * mask[..., None]
        return x

    def forward_with_cfg(self, x, t, dataset_idx, spacegroup, mask, cfg_scale, x_sc=None):
        """Forward pass of DiT, but also batches the unconditional forward pass for classifier-free
        guidance.

        Assumes batch x's and class labels are ordered such that the first half are the conditional
        samples and the second half are the unconditional samples.
        """
        half_x = x[: len(x) // 2]
        combined_x = torch.cat([half_x, half_x], dim=0)
        model_out = self.forward(combined_x, t, dataset_idx, spacegroup, mask, x_sc)

        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps
