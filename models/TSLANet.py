import torch
import torch.nn as nn
from einops import rearrange

"""
TSLANet: A Lightweight Adaptive Network for Time Series (ICML 2024).
Paper link: https://arxiv.org/abs/2404.08472
Official repo: https://github.com/emadeldeen24/TSLANet

TSLANet interleaves an Adaptive Spectral Block (ASB), which learns a
frequency-domain filter with an adaptive high-frequency mask, and an
Interactive Convolution Block (ICB) over patch embeddings. Adapted (and made
self-contained, dropping the timm / lightning dependencies) to the
Time-Series-Library long-term-forecasting interface.
"""


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Stochastic depth per sample (identical to timm's drop_path)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        nn.init.trunc_normal_(self.complex_weight_high, std=.02)
        nn.init.trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy and compute the median
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        adaptive_mask = ((normalized_energy > self.threshold_param).float()
                         - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)
        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # FFT along the patch dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted = x_weighted + x_weighted2

        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, C)
        return x


class TSLANet_layer(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_icb=True, use_asb=True,
                 adaptive_filter=True):
        super().__init__()
        self.use_icb = use_icb
        self.use_asb = use_asb
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim, adaptive_filter=adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        if self.use_icb and self.use_asb:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        emb_dim = getattr(configs, 'd_model', 128)
        depth = getattr(configs, 'e_layers', 3)
        dropout = getattr(configs, 'dropout', 0.15)
        self.patch_size = getattr(configs, 'patch_size', 16)
        # patch size must not exceed the look-back window
        self.patch_size = min(self.patch_size, self.seq_len)
        self.stride = max(self.patch_size // 2, 1)
        use_icb = bool(getattr(configs, 'ICB', True))
        use_asb = bool(getattr(configs, 'ASB', True))
        adaptive_filter = bool(getattr(configs, 'adaptive_filter', True))

        num_patches = int((self.seq_len - self.patch_size) / self.stride + 1)

        self.input_layer = nn.Linear(self.patch_size, emb_dim)

        dpr = [x.item() for x in torch.linspace(0, dropout, depth)]  # stochastic depth decay
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=emb_dim, drop=dropout, drop_path=dpr[i],
                          use_icb=use_icb, use_asb=use_asb,
                          adaptive_filter=adaptive_filter)
            for i in range(depth)
        ])

        self.out_layer = nn.Linear(emb_dim * num_patches, self.pred_len)

    def forecast(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev

        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        x = self.input_layer(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        outputs = self.out_layer(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means
        return outputs

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
