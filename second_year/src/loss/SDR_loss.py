
import numpy as np
from itertools import permutations
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

import scipy,time,numpy
import itertools

import torch

EPS = 1e-8
# class SingleSrcNegSDR(_Loss):
#     r"""Base class for single-source negative SI-SDR, SD-SDR and SNR.

#     Args:
#         sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
#             SI-SDR and ``sdsdr`` for SD-SDR [1].
#         zero_mean (bool, optional): by default it zero mean the target and
#             estimate before computing the loss.
#         take_log (bool, optional): by default the log10 of sdr is returned.
#         reduction (string, optional): Specifies the reduction to apply to
#             the output:
#             ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
#             ``'mean'``: the sum of the output will be divided by the number of
#             elements in the output.

#     Shape:
#         - est_targets : :math:`(batch, time)`.
#         - targets: :math:`(batch, time)`.

#     Returns:
#         :class:`torch.Tensor`: with shape :math:`(batch)` if ``reduction='none'`` else
#         [] scalar if ``reduction='mean'``.

#     Examples
#         >>> import torch
#         >>> from asteroid.losses import PITLossWrapper
#         >>> targets = torch.randn(10, 2, 32000)
#         >>> est_targets = torch.randn(10, 2, 32000)
#         >>> loss_func = PITLossWrapper(SingleSrcNegSDR("sisdr"),
#         >>>                            pit_from='pw_pt')
#         >>> loss = loss_func(est_targets, targets)

#     References
#         [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
#         International Conference on Acoustics, Speech and Signal
#         Processing (ICASSP) 2019.
#     """

#     def __init__(self, sdr_type, zero_mean=True, take_log=True, reduction="none", EPS=1e-8):
#         assert reduction != "sum", NotImplementedError
#         super().__init__(reduction=reduction)

#         assert sdr_type in ["snr", "sisdr", "sdsdr"]
#         self.sdr_type = sdr_type
#         self.zero_mean = zero_mean
#         self.take_log = take_log
#         self.EPS = 1e-8

#     def forward(self, est_target, target):
#         if target.size() != est_target.size() or target.ndim != 2:
#             raise TypeError(
#                 f"Inputs must be of shape [batch, time], got {target.size()} and {est_target.size()} instead"
#             )
#         # Step 1. Zero-mean norm
#         if self.zero_mean:
#             mean_source = torch.mean(target, dim=1, keepdim=True)
#             mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
#             target = target - mean_source
#             est_target = est_target - mean_estimate
#         # Step 2. Pair-wise SI-SDR.
#         if self.sdr_type in ["sisdr", "sdsdr"]:
#             # [batch, 1]
#             dot = torch.sum(est_target * target, dim=1, keepdim=True)
#             # [batch, 1]
#             s_target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + self.EPS
#             # [batch, time]
#             scaled_target = dot * target / s_target_energy
#         else:
#             # [batch, time]
#             scaled_target = target
#         if self.sdr_type in ["sdsdr", "snr"]:
#             e_noise = est_target - target
#         else:
#             e_noise = est_target - scaled_target
#         # [batch]
#         losses = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + self.EPS)
#         if self.take_log:
#             losses = 10 * torch.log10(losses + self.EPS)
#         losses = losses.mean() if self.reduction == "mean" else losses
#         return -losses

class SDR_loss(_Loss):
    def __init__(self, sdr_type, zero_mean=True, take_log=True, reduction="none", EPS=1e-8, clipping=30):
        super(SDR_loss, self).__init__()
        self.reduction=reduction
        self.EPS=float(EPS)
        self.clipping=10**(-1*clipping/10)
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
   

    def forward(self, est_target, target):
        if target.size() != est_target.size() or target.ndim != 2:
            raise TypeError(
                f"Inputs must be of shape [batch, time], got {target.size()} and {est_target.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(target, dim=1, keepdim=True)
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, 1]
            dot = torch.sum(est_target * target, dim=1, keepdim=True)
            # [batch, 1]
        
            s_target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + self.EPS
            # [batch, time]
            scaled_target = dot * target / s_target_energy
        else:
            # [batch, time]
            scaled_target = target

        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = est_target - target
        else:
            e_noise = est_target - scaled_target
        # [batch]
        scaled_target_power=torch.sum(scaled_target ** 2, dim=1)
        
        losses =  scaled_target_power/ (torch.sum(e_noise ** 2, dim=1) + self.EPS+self.clipping*scaled_target_power)#*scaled_target_power)

        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)
        losses = losses.mean() if self.reduction == "mean" else losses

        return -losses