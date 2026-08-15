"""PRF neuron and Spiking (SD-TCM) blocks.

Implements the Parallel Resonate-and-Fire (PRF) neuron and the Spike-Driven
Temporal and Channel Mixer (SD-TCM) blocks used in the paper:

    "PRF: Parallel Resonate and Fire Neuron for Long Sequence Learning in
    Spiking Neural Networks".

Neuron models:
  - PRFNeuron:        PRF neuron (complex-domain resonance, no explicit reset).
  - PRFNeuronReset:   PRF with an explicit reset (ablation, paper Table 9).
  - PRFNeuronTrain:   PRF variant with a trainable firing-rate scale.
  - DecoupledResetLIF: parallel LIF with the decoupled-reset training trick.

Blocks:
  - SpikingBlockGate: SD-TCM block used in the paper's experiments
    (``temporal`` selects the temporal neuron, ``spatial`` the spatial neuron).
"""
import math
from functools import partial
from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based.surrogate import heaviside

from src.models.sequence import SequenceModule
from src.models.sequence.kernels import registry as kernel_registry
from src.models.nn import LinearActivation, Activation, DropoutNd, Normalization
import src.utils.train
import src.utils as utils
from src.models.sequence.modules.bhrf_neuron import BHRFCell
from src.models.sequence.modules.elm_neuron import ELM
from src.models.sequence.modules.neuron import TCLIFNode, TSLIFNode, MaskedSlidingPSN, IFNode5PorderMaskD, IFNode5
from src.models.sequence.modules.para_base import ParaLIF


log = src.utils.train.get_logger(__name__)

contract = torch.einsum


def multiple_axis_slice(x, L):
    """
    x: (..., L1, L2, .., Lk)
    L: list of length k [l1, l2, .., lk]
    returns: x[..., :l1, :l2, .., :lk]
    """
    # TODO I don't see a way to do this programmatically in Pytorch without sacrificing speed so...
    assert len(L) > 0
    if len(L) == 1:
        return x[..., :L[0]]
    elif len(L) == 2:
        return x[..., :L[0], :L[1]]
    elif len(L) == 3:
        return x[..., :L[0], :L[1], :L[2]]
    elif len(L) == 4:
        return x[..., :L[0], :L[1], :L[2], :L[3]]
    else:
        raise NotImplementedError("lol")


class PRFNeuron(LIFNode):
    """Parallel Resonate-and-Fire (PRF) neuron.

    The membrane potential evolves in the complex domain:

        z_t = beta * z_{t-1} + dt * x_t,   beta = exp(dt * (-1/tau + i * theta))

    where ``log_dt`` and ``theta_log`` are learnable per-channel parameters. The
    real part of ``z_t`` drives the spike; there is no explicit reset (soft reset
    is equivalent to the constant threshold, see the paper).

    Training uses an FFT-based parallel scan over time (no sequential unroll);
    ``sequential_step`` provides the equivalent recurrent form for reference.
    """

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

        # Per-channel decay step dt ~ LogUniform(dt_min, dt_max)
        log_dt = torch.rand(channels) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        self.channels = channels

        # Per-channel resonance frequency theta ~ Uniform(0, 2*pi / fr_scale)
        max_phase = 2 * torch.pi
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
        s_seq = self.parallelization_step(x)
        return s_seq

    @staticmethod
    @torch.jit.script
    def sequential_step(x: torch.Tensor, log_dt: torch.Tensor, tau: torch.Tensor, theta_log: torch.Tensor,
                        v_threshold: torch.Tensor) -> torch.Tensor:
        """Recurrent (reference) form of the PRF dynamics."""
        dt = log_dt.exp()
        theta = theta_log.exp()
        a = (dt * (-1 / tau)).exp()
        alpha_real = a * (dt * theta).cos()
        alpha_img = a * (dt * theta).sin()

        # the input with T dimension
        T = x.shape[0]
        u_pre = torch.zeros_like(x[0])
        r_pre = torch.zeros_like(x[0])
        seq_list = []
        for t in range(T):
            # recurrent step
            u = alpha_real * u_pre - alpha_img * r_pre + dt * x[t]
            r = alpha_img * u_pre + alpha_real * r_pre
            spike = (u - v_threshold) >= 0
            u_pre = u
            r_pre = r
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

        soft_reset_threshold = self.v_threshold
        s_seq = self.surrogate_function(u_seq.real - soft_reset_threshold)
        return s_seq

    ### Parallel Process ###
    def charge(self, kernel, input_seq):
        T, D = kernel.shape

        if len(input_seq.shape) == 3:
            kernel_expand = kernel.squeeze().view(T, 1, D).contiguous()
        elif len(input_seq.shape) == 4:
            kernel_expand = kernel.squeeze().view(T, 1, D, 1).contiguous()
        else:
            raise NotImplementedError

        u_seq = self.conv_op(kernel_expand, input_seq, T)
        return u_seq

    @staticmethod
    @torch.jit.script  # error when complex kernel
    def conv_op(kernel_expand: torch.Tensor, input_seq: torch.Tensor, T: int) -> torch.Tensor:
        """Causal convolution of the kernel with the input via FFT."""
        output_fft = torch.fft.ifft(
            torch.fft.fft(kernel_expand, n=2 * T, dim=0)
            * torch.fft.fft(input_seq, n=2 * T, dim=0)
            ,
            n=2 * T, dim=0)
        u_seq = output_fft[:T]
        return u_seq.real

    @staticmethod
    def scan_kernel(beta: torch.Tensor, input_beta: torch.Tensor, timestep: int):
        """Build the convolution kernel K[t] = beta**t * input_beta."""
        K = beta.unsqueeze(-1) ** torch.arange(timestep, device=beta.device)  # (D L)
        B = input_beta.unsqueeze(-1)  # (D, 1)
        return (K * B).T

    @staticmethod
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


class PRFNeuronReset(LIFNode):
    """PRF neuron with an explicit (soft) reset.

    Same complex-domain dynamics as :class:`PRFNeuron`, but the firing
    threshold accumulates past spikes (``scan_dynamic_threshold``), which
    is equivalent to a soft reset. Used for the PRF+reset ablation
    (paper Table 9).
    """

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
        s_seq = self.parallelization_step(x)
        return s_seq

    @staticmethod
    @torch.jit.script
    def seqential_step(x: torch.Tensor, v_real, v_img, delta, v_threshold):
        # the input without T dimension
        v = (v_real + 1j * v_img) + delta * x

        v_real = v.real
        v_img = v.imag

        spike = heaviside(v_real - v_threshold)
        v_real = v_real - spike * v_threshold
        return spike, v_real, v_img

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

        soft_reset_threshold = self.v_threshold
        vth = self.scan_dynamic_threshold(self.v_threshold, beta.real, u_seq.real, x.shape[0])
        s_seq = self.surrogate_function(u_seq.real - vth)
        return s_seq

    def charge(self, kernel, input_seq):
        T, D = kernel.shape

        if len(input_seq.shape) == 3:
            kernel_expand = kernel.squeeze().view(T, 1, D).contiguous()
        elif len(input_seq.shape) == 4:
            kernel_expand = kernel.squeeze().view(T, 1, D, 1).contiguous()
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
    def scan_kernel(beta: torch.Tensor, input_beta: torch.Tensor, timestep: int):
        K = beta.unsqueeze(-1) ** torch.arange(timestep, device=beta.device)  # (D L)
        B = input_beta.unsqueeze(-1)  # (D, 1)
        return (K * B).T

    @staticmethod
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

    @staticmethod
    @torch.jit.script
    def scan_dynamic_threshold(threshold: torch.Tensor, beta: torch.Tensor, u_list: torch.Tensor,
                               timestep: int) -> torch.Tensor:
        final = []
        current_v = torch.ones_like(u_list[0]) * threshold
        vth_bias = torch.zeros_like(u_list[0])
        for i in range(timestep):
            final.append(current_v)

            fire_mask = torch.where(u_list[i] >= current_v, 1., 0.)
            vth_bias += threshold * fire_mask

            vth_bias *= beta
            current_v = threshold + vth_bias
        return torch.stack(final)


class PRFNeuronTrain(LIFNode):
    """PRF neuron with an additional trainable per-channel scale ``gamma``.

    Used for the sMNIST/psMNIST/seqCIFAR small models in the paper
    (``model.layer.temporal="prf_train"``).
    """
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

        self.register_memory('selective_vth', 0.)



        max_phase = 2 * torch.pi
        neuron_lr = kwargs.get('neuron_lr', 0.001)

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
        self.register("gamma", torch.log(torch.ones(channels) / torch.Tensor([tau])).cuda(), neuron_lr)

        self.decay_input = decay_input
        self.v_threshold = torch.Tensor([v_threshold]).cuda()
        self.v_threshold_float = v_threshold

    def forward(self, x):
        s_seq = self.parallelization_step(x)
        return s_seq

    @staticmethod
    @torch.jit.script
    def seqential_step(x: torch.Tensor, v_real, v_img, delta, v_threshold):
        # the input without T dimension
        v = (v_real + 1j * v_img) + delta * x

        v_real = v.real
        v_img = v.imag

        spike = heaviside(v_real - v_threshold)
        v_real = v_real - spike * v_threshold
        return spike, v_real, v_img

    def parallelization_step(self, x):
        dt = torch.exp(self.log_dt)
        theta = torch.exp(self.theta_log)

        if self.train_scale:
            theta = theta * self.fr_scale.sigmoid()

        beta = torch.exp(dt * (-torch.exp(self.gamma) + 1j * theta))

        input_beta = dt

        time_step = x.shape[0]
        x_seq = x
        kernel = self.scan_kernel(beta=beta, input_beta=input_beta, timestep=time_step)

        u_seq = self.charge(kernel=kernel, input_seq=x_seq)

        soft_reset_threshold = self.v_threshold + self.selective_vth
        s_seq = self.surrogate_function(u_seq.real - soft_reset_threshold)
        return s_seq

    def charge(self, kernel, input_seq):
        T, D = kernel.shape

        if len(input_seq.shape) == 3:
            kernel_expand = kernel.squeeze().view(T, 1, D).contiguous()
        elif len(input_seq.shape) == 4:
            kernel_expand = kernel.squeeze().view(T, 1, D, 1).contiguous()
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
    def scan_kernel(beta: torch.Tensor, input_beta: torch.Tensor, timestep: int):
        K = beta.unsqueeze(-1) ** torch.arange(timestep, device=beta.device)  # (D L)
        B = input_beta.unsqueeze(-1)  # (D, 1)
        return (K * B).T

    @staticmethod
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


class DecoupledResetLIF(LIFNode):
    """Parallel LIF trained with the decoupled-reset method.

    The soft reset is folded into an equivalent threshold scan, so the
    membrane potentials of all timesteps are computed in parallel
    during training while remaining equivalent to the sequential LIF.
    """
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = surrogate.ATan(),
                 detach_reset: bool = False, step_mode='m', backend='torch',
                 store_v_seq: bool = False, state_dim=None,
                 **kwargs):

        assert isinstance(tau, float) and tau > 1.
        assert v_reset == None

        step_mode = 'm'
        backend = 'torch'

        super().__init__(tau, decay_input, v_threshold, v_reset,
                         surrogate_function, detach_reset, step_mode, backend, store_v_seq, )
        self.beta = torch.as_tensor(1 - 1 / tau).cuda()

        self.decay_input = decay_input
        self.v_threshold = torch.Tensor([v_threshold]).cuda()

    def forward(self, x):
        s_seq = self.parallelization_step(x)
        return s_seq

    def seqential_step(self, x: torch.Tensor):
        return self.single_step_forward(x)

    def parallelization_step(self, x):
        beta = self.beta
        if self.decay_input:
            input_beta = 1 - beta
        else:
            input_beta = torch.ones_like(beta)

        time_step = x.shape[0]
        x_seq = x
        kernel = self.scan_kernel(beta=beta, input_beta=input_beta, timestep=time_step)
        u_seq = self.charge(kernel=kernel, input_seq=x_seq)
        soft_reset_threshold = self.scan_dynamic_threshold(threshold=self.v_threshold,
                                                           beta=beta,
                                                           u_list=u_seq,
                                                           timestep=time_step)
        s_seq = self.surrogate_function(u_seq - soft_reset_threshold)
        return s_seq

    ### Parallel Process ###
    def charge(self, kernel, input_seq):
        T = input_seq.shape[0]

        if len(input_seq.shape) == 2:
            kernel_expand = kernel.squeeze()[:, None]
        elif len(input_seq.shape) == 3:
            kernel_expand = kernel.squeeze()[:, None, None]
        elif len(input_seq.shape) == 4:
            kernel_expand = kernel.squeeze()[:, None, None, None]
        else:
            raise NotImplementedError

        kernel_expand = kernel_expand.expand_as(input_seq)

        u_seq = self.conv_op(kernel_expand, input_seq, T)
        return torch.as_tensor(u_seq).cuda()

    @staticmethod
    def conv_op(kernel_expand: torch.Tensor, input_seq: torch.Tensor, T: int) -> torch.Tensor:

        output_rfft = torch.fft.irfft(
            torch.fft.rfft(kernel_expand, n=2 * T, dim=0) *
            torch.fft.rfft(input_seq, n=2 * T, dim=0), n=2 * T, dim=0)
        u_seq = output_rfft[:T].real
        return u_seq

    @staticmethod
    @torch.jit.script
    def scan_kernel(beta: torch.Tensor, input_beta: torch.Tensor, timestep: int) -> torch.Tensor:
        final = []
        for i in range(timestep):
            final.append((beta ** i) * input_beta)
        return torch.stack(final)

    @staticmethod
    @torch.jit.script
    def scan_dynamic_threshold(threshold: torch.Tensor, beta: torch.Tensor, u_list: torch.Tensor,
                               timestep: int) -> torch.Tensor:
        final = []
        current_v = torch.ones_like(u_list[0]) * threshold
        vth_bias = torch.zeros_like(u_list[0])
        for i in range(timestep):
            final.append(current_v)

            fire_mask = torch.where(u_list[i] >= current_v, 1., 0.)
            vth_bias += threshold * fire_mask

            vth_bias *= beta
            current_v = threshold + vth_bias
        return torch.stack(final)


def Ter(x: torch.Tensor, alpha):
    pos = torch.where(x >= alpha, 1, 0)
    neg = torch.where(x <= alpha, -1, 0)
    return pos + neg


class SpikingBlock(SequenceModule):
    """Residual block stacking a PRF temporal neuron and a pointwise
    spatial neuron (predecessor of :class:`SpikingBlockGate`)."""
    def __init__(self, d_model, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.d_output = d_model
        self.transposed = transposed

        self.use_gsu = kernel_args.get('use_gsu', False)
        norm = kernel_args.get('norm', "batch")


        self.neuron1 = PRFNeuron(channels=self.h,
                                           fr_scale=kernel_args['fr_scale'],
                                           dt_min=kernel_args['dt_min'],
                                           dt_max=kernel_args['dt_max'])
        self.neuron2 = surrogate.ATan()
        self.neuron_gate = surrogate.ATan()

        dropout_fn = DropoutNd
        self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        self.dropout2 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        self.pro_linear1 = nn.Linear(self.h, self.h)
        self.pro_linear2 = nn.Linear(self.h, self.h)


        if isinstance(norm, str):
            self.norm = Normalization(self.d_output, transposed=self.transposed, _name_=norm)
        else:
            self.norm = Normalization(self.d_output, transposed=self.transposed, _name_='none')

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        u = u.permute(1, 0, 2).contiguous()
        # make sure the dimensions is corresponding with training frameworks (B, L, H) -> (L, B, H)
        """ Input and output shape (L, B, H) """

        s = self.dropout1(self.neuron1(u))  # (L B H)
        y = self.pro_linear1(s)

        x = u + y

        s = self.dropout2(self.neuron2(x - 0.5))
        y = self.pro_linear2(s) + x
        """ Input and output shape (L, B, H) """
        y = y.permute(1, 0, 2).contiguous()  # (L, B, H) -> (B, L, H)
        return y, None




class spikeGate(nn.Module):
    """Ternary gated activation {-1, 0, 1} with a trainable scale."""

    def __init__(self, alpha=1.):
        super().__init__()
        self.pos_spike = surrogate.ATan()
        self.neg_spike = surrogate.ATan()
        w = torch.Tensor([alpha])
        self.register("log_w", -torch.log(w), 0.001)

    def forward(self, x):
        th = torch.max(torch.abs(x)) * 0.15
        alpha = torch.exp(self.log_w)  # trainable

        out_pos = self.pos_spike(x - th)  # 1 if x >= vth
        out_neg = - self.neg_spike(-x - th)  # -1 if x <= -vth
        # {-1, 0, 1} * alpha
        return alpha * (out_pos + out_neg)

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class SpikingFFN(nn.Module):
    """Spiking feed-forward (channel-mixing) module."""
    def __init__(self, d_model, **kernel_args):
        super().__init__()
        self.h = d_model
        self.d_output = d_model

        self.neuron1 = spikeGate()
        self.neuron2 = spikeGate()

        self.pro_linear1 = nn.Linear(self.h, self.h)
        self.pro_linear2 = nn.Linear(self.h, self.h)

    def forward(self, x):
        x = self.neuron1(x)
        x = self.pro_linear1(x)
        x = self.neuron2(x)
        x = self.pro_linear2(x)
        return x


class GSU(nn.Module):
    """Gated Spiking Unit: channel-mixing with a spiking gate."""
    def __init__(self, d_model, **kernel_args):
        super().__init__()
        self.bias = kernel_args.get('bias', True)

        self.h = d_model
        self.d_output = d_model

        self.ternay1 = spikeGate()
        self.ternay2 = spikeGate()

        self.pro_linear1 = nn.Linear(self.h, self.h, bias=self.bias)
        self.pro_linear2 = nn.Linear(self.h, self.h, bias=self.bias)

    def forward(self, x):
        if self.bias:
            bias1 = self.pro_linear1.bias
            bias2 = self.pro_linear2.bias
        else:
            bias1 = None
            bias2 = None
        x1 = torch.nn.functional.linear(self.ternay1(x), self.pro_linear1.weight, bias1)
        x2 = torch.nn.functional.linear(x, self.ternay2(self.pro_linear2.weight), bias2)
        return x1 * x2


class SpikingBlockGate(SequenceModule):
    """SD-TCM block (Spike-Driven Temporal and Channel Mixer).

    ``temporal`` selects the temporal neuron (``None``/``"prf"`` = PRF,
    ``"prf_train"``, ``"prf_reset"``, ``"decoupled_reset"``, ``"lif"``,
    ``"tclif"``, ``"tslif"``, ``"ParaLIF"``, ``"bhrf"``, ``"elm"``,
    ``"psn"``/``"mpsn"``/``"spsn"``, ``"identity"``); ``spatial`` selects
    the spatial neuron (``"Binary"`` spiking neuron, ``"identity"``, or
    ``None`` for a plain linear path).
    """
    def __init__(self, d_model, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.d_output = d_model
        self.transposed = transposed
        fr_scale = kernel_args.get('fr_scale', 1.)

        self.spatial = kernel_args.get('spatial', None)
        self.temporal = kernel_args.get('temporal', None)
        self.bidirectional = kernel_args.get('bidirectional', False)
        self.tau = kernel_args.get('tau', 2.)
        self.neuron_lr = kernel_args.get('neuron_lr', 0.001)

        self.save_firerate = kernel_args.get('save_firerate', False)

        if self.save_firerate:
            self.TN_fr = 0.  # token mixing
            self.SN_fr = 0.  # channel mixing
            self.count = 0.

        v_th = kernel_args.get('v_th', 1.)
        dropout_fn = DropoutNd

        if self.temporal == "identity":
            self.neuron1 = nn.Identity()
            self.pro_linear1 = nn.Identity()
            self.dropout1 = nn.Identity()
        elif self.temporal == "psn":
            self.neuron1 = IFNode5(T=2000, surrogate_function=surrogate.ATan())
            self.pro_linear1 = nn.Linear(self.h, self.h)
            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        elif self.temporal == "mpsn":
            self.neuron1 = IFNode5PorderMaskD(T=2000, surrogate_function=surrogate.ATan(), P=8)
            self.pro_linear1 = nn.Linear(self.h, self.h)
            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        elif self.temporal == "spsn":
            self.neuron1 = MaskedSlidingPSN(order=32,
                                            surrogate_function=surrogate.ATan(),
                                            exp_init=True, backend='gemm')
            self.pro_linear1 = nn.Linear(self.h, self.h)
            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        elif self.temporal == "lif":
            self.neuron1 = LIFNode(tau=self.tau, decay_input=False, v_reset=None, detach_reset=True, step_mode='m',
                                   backend='torch')
            self.pro_linear1 = nn.Linear(self.h, self.h)
            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        elif self.temporal == "tclif":
            self.neuron1 = TCLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m', )
            self.pro_linear1 = nn.Linear(self.h, self.h)
            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        elif self.temporal == "tslif":
            self.neuron1 = TSLIFNode(surrogate_function=surrogate.ATan(), v_reset=None, dim=self.h, detach_reset=True,
                                     step_mode='m', )
            self.pro_linear1 = nn.Linear(self.h, self.h)
            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        elif self.temporal == "lif_cupy":
            self.neuron1 = LIFNode(tau=self.tau, decay_input=False, v_reset=None, detach_reset=True, step_mode='m',
                                   backend='cupy')
            self.pro_linear1 = nn.Linear(self.h, self.h)
            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        elif self.temporal == "decoupled_reset":
            self.neuron1 = DecoupledResetLIF(tau=self.tau)
            self.pro_linear1 = nn.Linear(self.h, self.h)
            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        elif self.temporal == "ParaLIF":
            self.neuron1 = ParaLIF(n_neuron=self.h, spike_mode="T")
            self.pro_linear1 = nn.Linear(self.h, self.h)
            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        elif self.temporal == "bhrf":
            self.neuron1 = BHRFCell(layer_size=self.h)
            self.pro_linear1 = nn.Linear(self.h, self.h)
            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        elif self.temporal == "elm":
            self.neuron1 = ELM(num_input=self.h, num_output=self.h)
            self.pro_linear1 = nn.Linear(self.h, self.h)
            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        elif self.temporal == "prf_reset":
            self.neuron1 = PRFNeuronReset(channels=self.h,
                                                     fr_scale=fr_scale,
                                                     dt_min=kernel_args['dt_min'],
                                                     dt_max=kernel_args['dt_max'],
                                                     v_threshold=v_th,
                                                     tau=self.tau,
                                                     neuron_lr=self.neuron_lr)

            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            if self.bidirectional:
                self.reverse_neuron1 = PRFNeuronReset(channels=self.h,
                                                                 fr_scale=fr_scale,
                                                                 dt_min=kernel_args['dt_min'],
                                                                 dt_max=kernel_args['dt_max'],
                                                                 v_threshold=v_th,
                                                                 tau=self.tau,
                                                                 neuron_lr=self.neuron_lr)
                self.pro_linear1 = nn.Linear(2 * self.h, self.h)
            else:
                self.pro_linear1 = nn.Linear(self.h, self.h)
        elif self.temporal == "prf_train":

            self.neuron1 = PRFNeuronTrain(channels=self.h,
                                                     fr_scale=fr_scale,
                                                     dt_min=kernel_args['dt_min'],
                                                     dt_max=kernel_args['dt_max'],
                                                     v_threshold=v_th,
                                                     tau=self.tau,
                                                     neuron_lr=self.neuron_lr)

            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            if self.bidirectional:
                self.reverse_neuron1 = PRFNeuronTrain(channels=self.h,
                                                                 fr_scale=fr_scale,
                                                                 dt_min=kernel_args['dt_min'],
                                                                 dt_max=kernel_args['dt_max'],
                                                                 v_threshold=v_th,
                                                                 tau=self.tau,
                                                                 neuron_lr=self.neuron_lr)
                self.pro_linear1 = nn.Linear(2 * self.h, self.h)
            else:
                self.pro_linear1 = nn.Linear(self.h, self.h)
        else:
            print("now use PRF neuron")

            self.neuron1 = PRFNeuron(channels=self.h,
                                               fr_scale=fr_scale,
                                               dt_min=kernel_args['dt_min'],
                                               dt_max=kernel_args['dt_max'],
                                               v_threshold=v_th,
                                               tau=self.tau,
                                               neuron_lr=self.neuron_lr)

            self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            if self.bidirectional:
                self.reverse_neuron1 = PRFNeuron(channels=self.h,
                                                           fr_scale=fr_scale,
                                                           dt_min=kernel_args['dt_min'],
                                                           dt_max=kernel_args['dt_max'],
                                                           v_threshold=v_th,
                                                           tau=self.tau,
                                                           neuron_lr=self.neuron_lr)
                self.pro_linear1 = nn.Linear(2 * self.h, self.h)
            else:
                self.pro_linear1 = nn.Linear(self.h, self.h)

        if self.temporal is not None:
            print(f"now use {self.temporal} neuron")


        if self.spatial == None:
            self.dropout2 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        elif self.spatial == "identity":
            self.pro_linear2 = nn.Identity()
        elif self.spatial == "Binary":
            self.neuron2 = surrogate.ATan()
            alpha = torch.log(torch.ones(1))
            self.register("alpha", alpha, self.neuron_lr)
            self.dropout2 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            self.pro_linear2 = nn.Linear(self.h, self.h)
        elif self.spatial == "Binary_vec":
            self.neuron2 = surrogate.ATan()
            alpha = torch.log(torch.ones(self.h))
            self.register("alpha", alpha, self.neuron_lr)
            self.dropout2 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            self.pro_linear2 = nn.Linear(self.h, self.h)
        elif self.spatial == "Ternary":
            self.neuron2 = spikeGate()
            self.dropout2 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            self.pro_linear2 = nn.Linear(self.h, self.h)
        elif self.spatial == "FFN":
            self.pro_ffn = SpikingFFN(self.h)
        elif self.spatial == "GSU":
            self.pro_gsu = GSU(self.h)
        elif self.spatial == "GLU":
            self.neuron2 = spikeGate()
            self.pro_linear = nn.Sequential(
                nn.Linear(self.h, 2 * self.h),
                nn.GLU(dim=-1),
            )
        else:
            raise NotImplementedError(f"spatial {self.spatial} not implemented yet")


    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask

        u = u.permute(1, 0, 2).contiguous()
        # make sure the dimensions is corresponding with training frameworks (B, L, H) -> (L, B, H)
        """ Input and output shape (L, B, H) """

        s = self.neuron1(u)  # (L B H)

        if self.bidirectional:
            s = torch.concat([s, self.reverse_neuron1(u.flip(dims=[0])).flip(dims=[0])], dim=-1)

        if self.save_firerate:
            self.TN_fr += s.detach().clone().mean()
            self.count += 1.  # for different batch

        y = self.pro_linear1(self.dropout1(s))

        x = u + y

        if self.spatial == None:
            y = self.dropout2(x)
        elif self.spatial == "identity":
            y = self.pro_linear2(x)
        elif self.spatial == "Binary":
            s = self.neuron2(x - 0.5)
            s_out = self.dropout2(s * torch.exp(self.alpha))  # {0, 1} * trainable alpha
            y = self.pro_linear2(s_out) + x
        elif self.spatial == "Binary_vec":
            s = self.dropout2(self.neuron2(x - 0.5) * torch.exp(self.alpha))  # {0, 1} * trainable alpha
            y = self.pro_linear2(s) + x
        elif self.spatial == "Tenary":
            s = self.dropout2(self.neuron2(x))  # {-1, 0, 1} * trainable alpha
            y = self.pro_linear2(s) + x
        elif self.spatial == "FFN":
            y = self.pro_ffn(s) + x
        elif self.spatial == "GSU":
            y = self.pro_gsu(x) + x
        else:
            raise NotImplementedError

        if self.save_firerate:
            self.SN_fr += s.detach().clone().mean()

        """ Input and output shape (L, B, H) """
        y = y.permute(1, 0, 2).contiguous()  # (L, B, H) -> (B, L, H)

        return y, None


class SpikingBlockGateSelective(SequenceModule):
    """Variant of :class:`SpikingBlockGate` with a selective (trainable)
    threshold. Experimental; not used in the paper."""
    def __init__(self, d_model, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.d_output = d_model
        self.transposed = transposed

        self.spatial = kernel_args.get('spatial', None)
        self.temporal = kernel_args.get('temporal', None)
        self.bidirectional = kernel_args.get('bidirectional', False)
        self.tau = kernel_args.get('tau', 2.)
        self.neuron_lr = kernel_args.get('neuron_lr', 0.001)

        self.save_firerate = kernel_args.get('save_firerate', False)

        if self.save_firerate:
            self.TN_fr = 0.  # token mixing
            self.SN_fr = 0.  # channel mixing
            self.count = 0.

        v_th = kernel_args.get('v_th', 1.)


        self.neuron1 = PRFNeuronTrain(channels=self.h,
                                                 fr_scale=kernel_args['fr_scale'],
                                                 dt_min=kernel_args['dt_min'],
                                                 dt_max=kernel_args['dt_max'],
                                                 v_threshold=v_th,
                                                 tau=self.tau,
                                                 neuron_lr=self.neuron_lr)

        dropout_fn = DropoutNd
        self.dropout1 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        if self.bidirectional:
            self.reverse_neuron1 = PRFNeuronTrain(channels=self.h,
                                                             fr_scale=kernel_args['fr_scale'],
                                                             dt_min=kernel_args['dt_min'],
                                                             dt_max=kernel_args['dt_max'],
                                                             v_threshold=v_th,
                                                             tau=self.tau,
                                                             neuron_lr=self.neuron_lr)
            self.pro_linear1 = nn.Linear(2 * self.h, self.h)
            select_pro1 = torch.randn(2 * self.h) * v_th
        else:
            self.pro_linear1 = nn.Linear(self.h, self.h)
            select_pro1 = torch.randn(self.h) * v_th
        self.select_pro1 = nn.Parameter(select_pro1)

        if self.temporal == "train":
            alpha_temp = torch.log(torch.ones(1))
            self.register("alpha_temp", alpha_temp, self.neuron_lr)

        if self.spatial == None:
            self.neuron2 = surrogate.ATan()
            self.dropout2 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            self.pro_linear2 = nn.Linear(self.h, self.h)
        elif self.spatial == "Binary":
            self.neuron2 = surrogate.ATan()
            alpha = torch.log(torch.ones(1))
            self.register("alpha", alpha, self.neuron_lr)
            self.dropout2 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            self.pro_linear2 = nn.Linear(self.h, self.h)
        elif self.spatial == "Binary_vec":
            self.neuron2 = surrogate.ATan()
            alpha = torch.log(torch.ones(self.h))
            self.register("alpha", alpha, self.neuron_lr)
            self.dropout2 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            self.pro_linear2 = nn.Linear(self.h, self.h)
        elif self.spatial == "Ternary":
            self.neuron2 = spikeGate()
            self.dropout2 = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            self.pro_linear2 = nn.Linear(self.h, self.h)
        elif self.spatial == "FFN":
            self.pro_ffn = SpikingFFN(self.h)
        elif self.spatial == "GSU":
            self.pro_gsu = GSU(self.h)
        elif self.spatial == "GLU":
            self.neuron2 = spikeGate()
            self.pro_linear = nn.Sequential(
                nn.Linear(self.h, 2 * self.h),
                nn.GLU(dim=-1),
            )
        else:
            raise NotImplementedError


    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask

        u = u.permute(1, 0, 2).contiguous()
        # make sure the dimensions is corresponding with training frameworks (B, L, H) -> (L, B, H)
        """ Input and output shape (L, B, H) """


        s = self.neuron1(u)  # (L B H)

        if self.bidirectional:
            s = torch.concat([s, self.reverse_neuron1(u.flip(dims=[0])).flip(dims=[0])], dim=-1)

        if self.save_firerate:
            self.TN_fr += s.detach().clone().mean()
            self.count += 1.

        y = self.pro_linear1(self.dropout1(s))

        x = u + y

        if self.spatial == None:
            s = self.neuron2(x - 0.5)
            s_out = self.dropout2(s)
            y = self.pro_linear2(s_out) + x
        if self.spatial == "Binary":
            s = self.neuron2(x - 0.5)
            s_out = self.dropout2(s * torch.exp(self.alpha))  # {0, 1} * trainable alpha
            y = self.pro_linear2(s_out) + x
        if self.spatial == "Binary_vec":
            s = self.dropout2(self.neuron2(x - 0.5) * torch.exp(self.alpha))  # {0, 1} * trainable alpha
            y = self.pro_linear2(s) + x
        if self.spatial == "Tenary":
            s = self.dropout2(self.neuron2(x))  # {-1, 0, 1} * trainable alpha
            y = self.pro_linear2(s) + x
        elif self.spatial == "FFN":
            y = self.pro_ffn(s) + x
        elif self.spatial == "GSU":
            y = self.pro_gsu(x) + x

        if self.save_firerate:
            self.SN_fr += s.detach().clone().mean()

        """ Input and output shape (L, B, H) """
        y = y.permute(1, 0, 2).contiguous()  # (L, B, H) -> (B, L, H)

        return y, None
