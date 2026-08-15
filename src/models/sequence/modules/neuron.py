import math
from abc import abstractmethod
from typing import Callable

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate
from torch import nn
from spikingjelly.activation_based.base import MemoryModule as base_MemoryModule


class IFNode5(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.fc = nn.Linear(T, T)
        nn.init.constant_(self.fc.bias, -1)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, x_seq.flatten(1))
        spike = self.surrogate_function(h_seq)
        return spike.view(x_seq.shape)


# masked PSN
class MaskedSlidingPSN(nn.Module):
    def gen_gemm_weight(self, T: int):
        weight = torch.zeros([T, T], device=self.weight.device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.order)
            length = min(end - start, self.order)
            weight[i][start: end] = self.weight[self.order - length: self.order]

        return weight

    def __init__(self, order: int, surrogate_function, exp_init: bool, backend='gemm'):
        super().__init__()
        self.order = order
        self.backend = backend
        if self.backend == 'gemm':
            if exp_init:
                weight = torch.ones([order])
                for i in range(order - 2, -1, -1):
                    weight[i] = weight[i + 1] / 2.

                self.weight = nn.Parameter(weight)
            else:
                self.weight = torch.ones([1, order])
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                self.weight = nn.Parameter(self.weight[0])

            self.threshold = nn.Parameter(torch.as_tensor(-1.))
            self.surrogate_function = surrogate_function


        elif self.backend == 'conv':
            self.weight = torch.zeros([1, 1, order])
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.weight = nn.Parameter(self.weight)
            self.threshold = nn.Parameter(torch.as_tensor(1.))
            self.surrogate_function = surrogate_function

    def forward(self, x_seq: torch.Tensor):
        if self.backend == 'gemm':
            weight = self.gen_gemm_weight(x_seq.shape[0])
            h_seq = torch.addmm(self.threshold, weight, x_seq.flatten(1)).view(x_seq.shape)
            return self.surrogate_function(h_seq)

        elif self.backend == 'conv':
            # x_seq.shape = [T, N, *]
            x_seq_shape = x_seq.shape
            # [T, N, *] -> [T, N] -> [N, T] -> [N, 1, T]
            x_seq = x_seq.flatten(1).t().unsqueeze(1)
            x_seq = F.pad(x_seq, pad=(self.order - 1, 0))
            x_seq = F.conv1d(x_seq, self.weight, stride=1)
            x_seq = x_seq.squeeze(1).t().view(x_seq_shape)
            return self.surrogate_function(x_seq - self.threshold)
        else:
            raise NotImplementedError(self.backend)


class LIFNR(nn.Module):
    # 去掉reset的plain if
    def __init__(self, tau: float, surrogate_function: surrogate.SurrogateFunctionBase):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.v_th = 1.
        self.tau = tau

    @staticmethod
    @torch.jit.script
    def pre_forward(x_seq: torch.Tensor, tau: float):
        decay_a = 1. - 1. / tau
        decay_b = 1. / tau
        h_t = torch.zeros_like(x_seq[0])
        h_seq = []
        for t in range(x_seq.shape[0]):
            h_t = decay_a * h_t + decay_b * x_seq[t]
            h_seq.append(h_t)

        return torch.stack(h_seq)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = self.pre_forward(x_seq, self.tau)
        spike = self.surrogate_function(h_seq - self.v_th)
        return spike


class DecayMaskedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mask1 = torch.ones_like(self.weight.data)
        mask0 = torch.tril(mask1)
        self.register_buffer('mask0', mask0)
        self.register_buffer('mask1', mask1)
        self.k = 0.
        # k should be set as epoch / (epochs - 1)

    @staticmethod
    @torch.jit.script
    def gen_mask(k: float, mask0: torch.Tensor, mask1: torch.Tensor):
        return k * mask0 + (1. - k) * mask1

    @staticmethod
    @torch.jit.script
    def gen_masked_weight(weight: torch.Tensor, k: float, mask0: torch.Tensor, mask1: torch.Tensor):
        return weight * (k * mask0 + (1. - k) * mask1)

    def masked_weight(self):
        return self.gen_masked_weight(self.weight, self.k, self.mask0, self.mask1)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight * self.gen_mask(self.k, self.mask0, self.mask1), self.bias)


# sliding PSN
class IFNode5MaskD(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.fc = DecayMaskedLinear(T, T)
        nn.init.constant_(self.fc.bias, -1)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.masked_weight(), x_seq.flatten(1))
        spike = self.surrogate_function(h_seq)
        return spike.view(x_seq.shape)


class DecayPorderMaskedLinear(nn.Linear):
    def __init__(self, P: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P = P
        mask1 = torch.ones_like(self.weight.data)
        mask0 = torch.tril(mask1) * torch.triu(mask1, -(P - 1))
        self.register_buffer('mask0', mask0)
        self.register_buffer('mask1', mask1)
        self.k = 0.
        # k should be set as epoch / (epochs - 1)

    @staticmethod
    @torch.jit.script
    def gen_mask(k: float, mask0: torch.Tensor, mask1: torch.Tensor):
        return k * mask0 + (1. - k) * mask1

    @staticmethod
    @torch.jit.script
    def gen_masked_weight(weight: torch.Tensor, k: float, mask0: torch.Tensor, mask1: torch.Tensor):
        return weight * (k * mask0 + (1. - k) * mask1)

    def masked_weight(self):
        return self.gen_masked_weight(self.weight, self.k, self.mask0, self.mask1)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight * self.gen_mask(self.k, self.mask0, self.mask1), self.bias)


class IFNode5PorderMaskD(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase, P: int):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.fc = DecayPorderMaskedLinear(P, T, T)
        nn.init.constant_(self.fc.bias, -1)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.masked_weight(), x_seq.flatten(1))
        spike = self.surrogate_function(h_seq)
        return spike.view(x_seq.shape)


class BaseNode(base_MemoryModule):
    def __init__(self,
                 v_threshold: float = 1.,
                 v_reset: float = 0.,
                 surrogate_function: Callable = None,
                 detach_reset: bool = False,
                 step_mode='s', backend='torch',
                 store_v_seq: bool = False):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold

        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class TCLIFNode(BaseNode):
    def __init__(self,
                 v_threshold=1.,
                 v_reset=0.,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 hard_reset=False,
                 step_mode='m',
                 k=2,
                 decay_factor: torch.Tensor = torch.full([1, 2], 0, dtype=torch.float),
                 gamma: float = 0.5):
        super(TCLIFNode, self).__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.k = k
        for i in range(1, self.k + 1):
            self.register_memory('v' + str(i), 0.)

        self.names = self._memories
        self.hard_reset = hard_reset
        self.gamma = gamma
        self.decay = decay_factor
        self.decay_factor = torch.nn.Parameter(decay_factor)

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        # v1: membrane potential of dendritic compartment
        # v2: membrane potential of somatic compartment
        self.names['v1'] = self.names['v1'] - torch.sigmoid(self.decay_factor[0][0]) * self.names['v2'] + x
        self.names['v2'] = self.names['v2'] + torch.sigmoid(self.decay_factor[0][1]) * self.names['v1']
        self.v = self.names['v2']

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if not self.hard_reset:
            # soft reset
            self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.gamma)
            self.names['v2'] = self.jit_soft_reset(self.names['v2'], spike_d, self.v_threshold)
        else:
            # hard reset
            for i in range(2, self.k + 1):
                self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d, self.v_reset)

    def forward(self, x: torch.Tensor):
        # shape is L, B, D
        return super().multi_step_forward(x)

    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
               f"hard_reset={self.hard_reset}, " \
               f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"


class TSBaseNode(base_MemoryModule):
    def __init__(self,
                 v_threshold: float = 1.,
                 v_reset: float = 0.,
                 surrogate_function: Callable = None,
                 detach_reset: bool = False,
                 step_mode='s', backend='torch', dim=128,
                 store_v_seq: bool = True):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('v_s', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold

        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq

        self.alpha_s = torch.nn.Parameter(torch.randn([1, dim], dtype=torch.float))
        self.alpha_l = torch.nn.Parameter(torch.randn([1, dim], dtype=torch.float))

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset

        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        # return self.surrogate_function(self.v - self.v_threshold, 2.0)
        return self.surrogate_function(self.v - self.v_threshold)

    def sl_neuronal_fire(self):
        # s_s = self.surrogate_function(self.v - self.v_threshold, 2.0)
        # s_l = self.surrogate_function(self.v_s - self.v_threshold,  2.0)
        s_s = self.surrogate_function(self.v - self.v_threshold)
        s_l = self.surrogate_function(self.v_s - self.v_threshold)
        return s_s, s_l

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        # spike = self.neuronal_fire()
        s_s, s_l = self.sl_neuronal_fire()
        spike = self.alpha_s * s_s + self.alpha_l * s_l
        # self.neuronal_reset(spike)
        self.neuronal_reset(s_s, s_l)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class TSLIFNode(TSBaseNode):
    def __init__(self,
                 v_threshold=1.0,
                 v_reset=0.,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 hard_reset=False,
                 step_mode='m',
                 k=2,
                 decay_factor: torch.Tensor = torch.tensor([0.8, 0.2, 0.3, 0.7], dtype=torch.float),
                 dim=128,
                 gamma: float = 0.5):
        super(TSLIFNode, self).__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, dim=dim)
        self.k = k
        for i in range(1, self.k + 1):
            self.register_memory('v' + str(i), 0.)

        self.names = self._memories
        self.hard_reset = hard_reset
        self.gamma = gamma
        self.decay_factor = torch.nn.Parameter(decay_factor)
        self.kk = torch.nn.Parameter(torch.tensor([0.8], dtype=torch.float))
        self.yy = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float))

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        # self.names['v1'] = self.names['v1'] - torch.sigmoid(self.decay_factor[0][0]) * self.names['v2'] + x
        # self.names['v2'] = self.names['v2'] + torch.sigmoid(self.decay_factor[0][1]) * self.names['v1']

        self.names['v1'] = self.decay_factor[0] * self.names['v1'] + self.decay_factor[1] * x - self.yy * self.names[
            'v2']
        self.names['v2'] = self.decay_factor[2] * self.names['v2'] + self.decay_factor[3] * x - self.kk * self.names[
            'v1']

        # self.names['v1'] =  self.names['v1'] + (1 - torch.sigmoid(self.decay_factor[0])) * x
        # self.names['v2'] =  self.names['v2'] + (1 - torch.sigmoid(self.decay_factor[1])) * x - self.names['v1']

        self.v = self.names['v2']
        self.v_s = self.names['v1']

    def neuronal_reset(self, spike_s, spike_l):

        if not self.hard_reset:
            # soft reset
            # self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.gamma)
            self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_l, self.gamma)
            self.names['v2'] = self.jit_soft_reset(self.names['v2'], spike_s, self.v_threshold)
        else:
            # hard reset
            for i in range(2, self.k + 1):
                self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d, self.v_reset)

    def forward(self, x: torch.Tensor):
        # self.v = 0.
        # self.v1 = 0.
        # self.v2 = 0.
        # shape is L, B, D
        return super().multi_step_forward(x)

    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
               f"hard_reset={self.hard_reset}, " \
               f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"


