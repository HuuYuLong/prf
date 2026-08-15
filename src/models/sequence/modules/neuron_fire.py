import math
from typing import Callable
import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate


class PRF(nn.Module):
    def __init__(self, channels, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = surrogate.ATan(),
                 detach_reset: bool = False, step_mode='m', backend='torch',
                 store_v_seq: bool = False, fr_scale: float = 1., dt_min=0.1, dt_max=0.001, train_scale: bool = False,
                 **kwargs):

        assert isinstance(tau, float) and tau > 1.
        assert v_reset == None
        assert channels != None

        step_mode = 'm'
        backend = 'torch'

        super().__init__(tau, decay_input, v_threshold, v_reset,
                         surrogate_function, detach_reset, step_mode, backend, store_v_seq, )
        neuron_lr = kwargs.get('neuron_lr', 0.001)
        max_phase = 2 * torch.pi

        log_dt = torch.rand(channels) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        self.channels = channels
        u2 = torch.rand(channels)  # uniform distribution
        self.train_scale = train_scale

        theta_log = torch.log(max_phase * u2 / fr_scale)
        self.fr_scale = fr_scale

        self.register("log_dt", log_dt, neuron_lr)
        self.register("theta_log", theta_log, neuron_lr)

        self.decay_input = decay_input
        self.v_threshold = torch.Tensor([v_threshold]).cuda()
        self.v_threshold_float = v_threshold

    def forward(self, x):
        if self.training:
            s_seq = self.parallelization_step(x)
        else:
            # print("now use sequential step for inference, please use parallel step for training!")
            # Ensure all arguments are tensors
            log_dt = torch.tensor(self.log_dt, device=x.device)
            tau = torch.tensor(self.tau, device=x.device)
            theta_log = torch.tensor(self.theta_log, device=x.device)
            v_threshold = torch.tensor(self.v_threshold, device=x.device)
            s_seq = self.sequential_step(x, log_dt, tau, theta_log, v_threshold)
        return s_seq

    @staticmethod
    @torch.jit.script
    def sequential_step(x: torch.Tensor, log_dt: torch.Tensor, tau: torch.Tensor, theta_log: torch.Tensor,
                        v_threshold: torch.Tensor) -> torch.Tensor:
        dt = log_dt.exp()
        theta = theta_log.exp()
        a = (dt * (-1 / tau)).exp()
        alpha_real = a * (dt * theta).cos()
        alpha_img = a * (dt * theta).sin()

        # the input with T dimension
        T = x.shape[0]
        u = torch.zeros_like(x[0])
        r = torch.zeros_like(x[0])
        seq_list = []
        for t in range(T):
            # recurrent step
            u = alpha_real * u - alpha_img * r + dt * x[t]
            r = alpha_img * u + alpha_real * r
            spike = (r - v_threshold) >= 0
            seq_list.append(spike)
        s_seq = torch.stack(seq_list).to(dtype=x.dtype)
        return s_seq

    def parallelization_step(self, x):
        dt = torch.exp(self.log_dt)
        theta = torch.exp(self.theta_log)

        if self.train_scale:
            theta = theta * self.fr_scale.sigmoid()

        beta = torch.exp(dt * (-1 / self.tau + 1j * theta))
        input_beta = dt
        time_step = x.shape[0]
        x_seq = x
        kernel = self.scan_kernel(beta=beta, input_beta=input_beta, timestep=time_step)
        u_seq = self.charge(kernel=kernel, input_seq=x_seq)
        s_seq = self.surrogate_function(u_seq.real - self.v_threshold)
        return s_seq

    ### Parallel Process ###
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

    @staticmethod
    # @torch.jit.script
    def scan_kernel_complex(beta: torch.Tensor, input_beta: torch.Tensor, timestep: int) -> torch.Tensor:
        seq = torch.empty(size=(timestep, beta.shape[0]), dtype=beta.dtype, device=beta.device)
        for i in range(timestep):
            seq[i] = ((beta ** i) * input_beta)
        return seq

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def extra_repr(self):
        return super().extra_repr() + f', train_fr_scale={self.train_scale}, channel={self.channels}, fr_scale={self.fr_scale}'
