import math
from typing import Callable

import torch
import torch.nn as nn

from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based.surrogate import heaviside

try:
    from flashfftconv import FlashFFTConv # https://github.com/HazyResearch/flash-fft-conv
    FLASH_FFT_FLAG = True
except:
    FLASH_FFT_FLAG = False


class ParallelResonateFire(LIFNode):
    def __init__(self, channels, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = surrogate.ATan(),
                 detach_reset: bool = False, step_mode='m', backend='torch',
                 store_v_seq: bool = False, fr_scale: float = 1., dt_min=0.1, dt_max=0.001, use_flash_fft= False,
                 **kwargs):

        assert isinstance(tau, float) and tau > 1.
        assert v_reset == None
        assert channels != None
        if use_flash_fft:
            assert FLASH_FFT_FLAG == True

        step_mode = 'm'
        backend = 'torch'

        super().__init__(tau, decay_input, v_threshold, v_reset,
                         surrogate_function, detach_reset, step_mode, backend, store_v_seq, )
        # self.fr_scale = fr_scale
        # self.channels = channels
        self.use_flash_fft = use_flash_fft
        if self.use_flash_fft:
            self.flash_fft = FlashFFTConv(32768, dtype=torch.bfloat16)  # generally more stable!

        max_phase = 2 * torch.pi

        log_dt = torch.rand(channels) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        u2 = torch.rand(channels)  # uniform distribution
        theta_log = torch.log(max_phase * u2 / fr_scale)

        self.register("log_dt", log_dt, 0.001)
        self.register("theta_log", theta_log, 0.001)

    def forward(self, x):
        # if self.training:
        s_seq = self.parallelization_step(x)
        # else:
        # sequential_step
        return s_seq

    @staticmethod
    @torch.jit.script
    def sequential_step(x: torch.Tensor, v, delta, tau, theta, v_threshold):
        # the input without T dimension
        v = torch.exp(delta * (-1 / tau + 1j * theta)) * v + delta * x
        spike = heaviside(v.real - v_threshold)
        return spike, v

    def parallelization_step(self, x):
        dt = torch.exp(self.log_dt)
        theta = torch.exp(self.theta_log)

        beta = torch.exp(dt * (-1 / self.tau + 1j * theta))
        input_beta = dt

        time_step = x.shape[0]
        x_seq = x
        kernel = self.scan_kernel(beta=beta, input_beta=input_beta, timestep=time_step)
        u_seq = self.charge(kernel=kernel, input_seq=x_seq)

        s_seq = self.surrogate_function(u_seq.real - self.v_threshold)
        return s_seq

    def charge(self, kernel, input_seq):
        T, D = kernel.shape

        if len(input_seq.shape) == 3:
            kernel_expand = kernel.squeeze().view(T, 1, D).contiguous()
            # kernel_expand_dendr = kernel_dendr.squeeze().view(T, 1, D).contiguous()
        elif len(input_seq.shape) == 4:
            kernel_expand = kernel.squeeze().view(T, 1, D, 1).contiguous()
            # kernel_expand_dendr = kernel_dendr.squeeze().view(T, 1, D, 1).contiguous()
        else:
            raise NotImplementedError

        if self.use_flash_fft:
            x = input_seq.permute(1, 2, 0).contiguous().to(dtype=torch.bfloat16)  # (L B H) -> (B H L)
            k = kernel.T.to(dtype=torch.float32)
            u_seq = self.flash_fft(x, k).permute(2, 0, 1).contiguous().to(dtype=torch.float32)
        else:
            u_seq = self.conv_op(kernel_expand, input_seq, T)
        return u_seq

    @staticmethod
    @torch.jit.script  # error when complex kernel
    def conv_op(kernel_expand: torch.Tensor, input_seq: torch.Tensor, T: int) -> torch.Tensor:
        output_fft = torch.fft.ifft(
            torch.fft.fft(kernel_expand, n=2 * T, dim=0)
            * torch.fft.fft(input_seq, n=2 * T, dim=0)
            ,
            n=2 * T, dim=0)
        u_seq = output_fft[:T]
        return u_seq.real

    @staticmethod
    # @torch.jit.script
    def scan_kernel(beta: torch.Tensor, input_beta: torch.Tensor, timestep: int):
        K = beta.unsqueeze(-1) ** torch.arange(timestep, device=beta.device)  # (D L)
        B = input_beta.unsqueeze(-1)  # (D, 1)
        return (K * B).T

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


if __name__ == '__main__':
    T = 128
    B = 64
    C = 256
    prf1 = ParallelResonateFire(1)
    prf2 = ParallelResonateFire(C)

    x1 = torch.rand((T,B,C))

    out1 = prf1(x1)
    out2 = prf2(x1)

    assert out1.shape == (T, B, C)
    assert out2.shape == (T, B, C)

