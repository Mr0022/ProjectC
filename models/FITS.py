import torch
import torch.nn as nn


class Model(nn.Module):
    """
    FITS: Frequency Interpolation Time Series Analysis (ICLR 2024)
    Paper link: https://arxiv.org/abs/2307.03756
    Official repo: https://github.com/VEWOXIC/FITS

    A near-parameter-free forecaster: it removes the instance mean/variance
    (RIN), keeps only the low-frequency band of the look-back window and learns
    a single complex-valued linear layer that interpolates that band up to the
    (look-back + horizon) length before mapping back to the time domain.

    Adapted to the Time-Series-Library long-term-forecasting interface.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = getattr(configs, 'individual', False)
        self.channels = configs.enc_in

        # Low-pass cut-off (number of retained frequency bins).
        # 0 -> pick a sensible default; always clamp to the rFFT length.
        base_freq = self.seq_len // 2 + 1
        cut_freq = getattr(configs, 'cut_freq', 0)
        if cut_freq <= 0:
            cut_freq = self.seq_len // 4 + 1
        self.dominance_freq = int(min(max(cut_freq, 1), base_freq))

        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len
        upsample_dim = int(self.dominance_freq * self.length_ratio)

        if self.individual:
            self.freq_upsampler = nn.ModuleList([
                nn.Linear(self.dominance_freq, upsample_dim).to(torch.cfloat)
                for _ in range(self.channels)
            ])
        else:
            # complex linear layer for frequency upsampling
            self.freq_upsampler = nn.Linear(
                self.dominance_freq, upsample_dim).to(torch.cfloat)

    def forecast(self, x):
        # RIN (reversible instance normalization)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq:] = 0  # low-pass filter
        low_specx = low_specx[:, 0:self.dominance_freq, :]

        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0),
                 int(self.dominance_freq * self.length_ratio),
                 low_specx.size(2)],
                dtype=low_specx.dtype, device=low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i])
        else:
            low_specxy_ = self.freq_upsampler(
                low_specx.permute(0, 2, 1)).permute(0, 2, 1)

        # zero padding up to the (seq_len + pred_len) spectrum
        low_specxy = torch.zeros(
            [low_specxy_.size(0),
             int((self.seq_len + self.pred_len) / 2 + 1),
             low_specxy_.size(2)],
            dtype=low_specxy_.dtype, device=low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_

        low_xy = torch.fft.irfft(low_specxy, n=self.seq_len + self.pred_len, dim=1)
        low_xy = low_xy * self.length_ratio  # energy compensation

        xy = low_xy * torch.sqrt(x_var) + x_mean  # reverse RIN
        return xy

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
