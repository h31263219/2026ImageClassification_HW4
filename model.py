"""PromptIR for blind image restoration.

Faithful re-implementation (from scratch — no pretrained weights) of:
  Potlapalli, V. et al. "PromptIR: Prompting for All-in-One Blind Image
  Restoration." NeurIPS 2023.
  https://arxiv.org/abs/2306.13090

The backbone follows Restormer (MDTA + GDFN transformer blocks in a
4-level U-Net). PromptIR adds Prompt Generation Blocks (PGB) inserted
in the decoder to inject degradation-adaptive prompts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- LayerNorm variants (channel-first) ----------

class BiasFreeLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-5

    def forward(self, x):
        # x: (B, C, H, W)
        sigma = x.var(dim=1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + self.eps) * self.weight.view(1, -1, 1, 1)


class WithBiasLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = 1e-5

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + self.eps) * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def make_norm(dim, kind="WithBias"):
    return WithBiasLayerNorm(dim) if kind == "WithBias" else BiasFreeLayerNorm(dim)


# ---------- Transformer block components ----------

class MDTA(nn.Module):
    """Multi-DConv Head Transposed (channel) Attention."""

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1,
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (B, h, d, d)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated DConv Feed-Forward Network."""

    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1,
                                padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False,
                 norm_kind="WithBias"):
        super().__init__()
        self.norm1 = make_norm(dim, norm_kind)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = make_norm(dim, norm_kind)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------- Up/Down sampling & patch embed ----------

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ---------- Prompt Generation Block ----------

class PromptGenBlock(nn.Module):
    """Learnable prompt components, weighted by per-image features, then
    interpolated to feature map size and refined by a 3x3 conv.
    """

    def __init__(self, prompt_dim, prompt_len=5, prompt_size=32, lin_dim=192):
        super().__init__()
        self.prompt_param = nn.Parameter(
            torch.randn(1, prompt_len, prompt_dim, prompt_size, prompt_size) * 0.02
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))  # (B, C)
        weights = F.softmax(self.linear_layer(emb), dim=-1)  # (B, L)
        # (B, L, 1, 1, 1) * (1, L, C, Hp, Wp) -> (B, L, C, Hp, Wp)
        prompt = weights.view(B, -1, 1, 1, 1) * self.prompt_param
        prompt = prompt.sum(dim=1)  # (B, C, Hp, Wp)
        prompt = F.interpolate(prompt, size=(H, W), mode="bilinear", align_corners=False)
        return self.conv3x3(prompt)


# ---------- Full PromptIR ----------

class PromptIR(nn.Module):
    """PromptIR with prompt blocks in the decoder.

    Args:
        dim: base channel count.
        num_blocks: transformer blocks at each of the 4 encoder/decoder levels.
        num_refinement_blocks: extra refinement blocks at output resolution.
        heads: attention heads per level.
        prompt_lens / prompt_dims / prompt_sizes: PGB hyper-parameters for the
            3 decoder insertion points (after latent, after decoder L3, after decoder L2).
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=36,
        num_blocks=(2, 3, 3, 4),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        norm_kind="WithBias",
        prompt_lens=(5, 5, 5),
        prompt_sizes=(64, 32, 16),
        use_prompt=True,
    ):
        super().__init__()

        self.use_prompt = use_prompt
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        def stack(level_dim, n_heads, n_blocks):
            return nn.Sequential(*[
                TransformerBlock(level_dim, n_heads, ffn_expansion_factor, bias, norm_kind)
                for _ in range(n_blocks)
            ])

        d1, d2, d3, d4 = dim, dim * 2, dim * 4, dim * 8
        self.encoder_level1 = stack(d1, heads[0], num_blocks[0])
        self.down1_2 = Downsample(d1)
        self.encoder_level2 = stack(d2, heads[1], num_blocks[1])
        self.down2_3 = Downsample(d2)
        self.encoder_level3 = stack(d3, heads[2], num_blocks[2])
        self.down3_4 = Downsample(d3)
        self.latent = stack(d4, heads[3], num_blocks[3])

        # Decoder transforms — note pre-reduction conv before each decoder stack
        self.up4_3 = Upsample(d4)
        self.reduce_chan_level3 = nn.Conv2d(d4, d3, kernel_size=1, bias=bias)
        self.decoder_level3 = stack(d3, heads[2], num_blocks[2])

        self.up3_2 = Upsample(d3)
        self.reduce_chan_level2 = nn.Conv2d(d3, d2, kernel_size=1, bias=bias)
        self.decoder_level2 = stack(d2, heads[1], num_blocks[1])

        self.up2_1 = Upsample(d2)
        # No reduce at level 1: concat of up2_1(d2 -> d1) with encoder_level1(d1) = 2*d1
        self.decoder_level1 = stack(d2, heads[0], num_blocks[0])

        self.refinement = stack(d2, heads[0], num_refinement_blocks)
        self.output = nn.Conv2d(d2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # Prompt blocks
        # Inserted at: (a) after latent (channels=d4), (b) after decoder_level3 (channels=d3),
        # (c) after decoder_level2 (channels=d2).
        # Each PGB outputs an additive feature stack that is concatenated with the
        # current feature map, then mixed by a transformer block and a 1x1 conv reduce.
        if self.use_prompt:
            p3_dim = max(8, d4 * 2 // 3)  # ~2/3 of latent channels
            p2_dim = max(8, d3 * 2 // 3)
            p1_dim = max(8, d2 * 2 // 3)

            self.prompt3 = PromptGenBlock(prompt_dim=p3_dim, prompt_len=prompt_lens[2],
                                          prompt_size=prompt_sizes[2], lin_dim=d4)
            self.prompt2 = PromptGenBlock(prompt_dim=p2_dim, prompt_len=prompt_lens[1],
                                          prompt_size=prompt_sizes[1], lin_dim=d3)
            self.prompt1 = PromptGenBlock(prompt_dim=p1_dim, prompt_len=prompt_lens[0],
                                          prompt_size=prompt_sizes[0], lin_dim=d2)

            self.noise_level3 = TransformerBlock(d4 + p3_dim, heads[3], ffn_expansion_factor, bias, norm_kind)
            self.reduce_noise3 = nn.Conv2d(d4 + p3_dim, d4, kernel_size=1, bias=bias)

            self.noise_level2 = TransformerBlock(d3 + p2_dim, heads[2], ffn_expansion_factor, bias, norm_kind)
            self.reduce_noise2 = nn.Conv2d(d3 + p2_dim, d3, kernel_size=1, bias=bias)

            self.noise_level1 = TransformerBlock(d2 + p1_dim, heads[1], ffn_expansion_factor, bias, norm_kind)
            self.reduce_noise1 = nn.Conv2d(d2 + p1_dim, d2, kernel_size=1, bias=bias)

    def forward(self, inp):
        # Encoder
        e1_in = self.patch_embed(inp)
        e1 = self.encoder_level1(e1_in)
        e2 = self.encoder_level2(self.down1_2(e1))
        e3 = self.encoder_level3(self.down2_3(e2))
        latent = self.latent(self.down3_4(e3))

        # Prompt @ latent
        if self.use_prompt:
            p3 = self.prompt3(latent)
            latent = self.reduce_noise3(self.noise_level3(torch.cat([latent, p3], dim=1)))

        # Decoder level 3
        d3 = self.up4_3(latent)
        d3 = self.reduce_chan_level3(torch.cat([d3, e3], dim=1))
        d3 = self.decoder_level3(d3)

        # Prompt @ d3
        if self.use_prompt:
            p2 = self.prompt2(d3)
            d3 = self.reduce_noise2(self.noise_level2(torch.cat([d3, p2], dim=1)))

        # Decoder level 2
        d2 = self.up3_2(d3)
        d2 = self.reduce_chan_level2(torch.cat([d2, e2], dim=1))
        d2 = self.decoder_level2(d2)

        # Prompt @ d2
        if self.use_prompt:
            p1 = self.prompt1(d2)
            d2 = self.reduce_noise1(self.noise_level1(torch.cat([d2, p1], dim=1)))

        # Decoder level 1
        d1 = self.up2_1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # channels = 2*dim, matches decoder_level1
        d1 = self.decoder_level1(d1)

        out = self.refinement(d1)
        out = self.output(out) + inp  # residual
        return out


def build_promptir(config: str = "light", use_prompt: bool = True) -> PromptIR:
    """Convenience builder. Supports 'light', 'medium', 'standard', 'large' configs.

    use_prompt=False degenerates the model to a pure Restormer (no PGBs) for
    ablation studies on the contribution of the prompt mechanism.
    """
    if config == "light":
        return PromptIR(dim=36, num_blocks=(2, 3, 3, 4), num_refinement_blocks=2,
                        heads=(1, 2, 4, 8), use_prompt=use_prompt)
    if config == "medium":
        # Same block counts as light, but wider channels (dim 36→48).
        # Trades ~2× per-block FFN compute for materially more capacity,
        # while keeping the deepest-level attention from blowing up.
        return PromptIR(dim=48, num_blocks=(2, 3, 3, 4), num_refinement_blocks=2,
                        heads=(1, 2, 4, 8), use_prompt=use_prompt)
    if config == "standard":
        return PromptIR(dim=48, num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                        heads=(1, 2, 4, 8), use_prompt=use_prompt)
    if config == "large":
        return PromptIR(dim=48, num_blocks=(4, 6, 6, 8), num_refinement_blocks=8,
                        heads=(1, 2, 4, 8), use_prompt=use_prompt)
    raise ValueError(f"Unknown config: {config}")


if __name__ == "__main__":
    # Quick sanity check
    model = build_promptir("light")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params / 1e6:.2f}M")
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print("Out:", y.shape)
