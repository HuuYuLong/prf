# https://github.com/AdaptiveAILab/brf-neurons
import torch
import math

# from .. import functional
# from .linear_layer import LinearMask
################################################################
# Neuron update functional
################################################################

DEFAULT_MASK_PROB = 0

TRAIN_B_offset = True
DEFAULT_RF_B_offset = 1.

# here: Depends on the initialization
DEFAULT_RF_ADAPTIVE_B_offset_a = 1
DEFAULT_RF_ADAPTIVE_B_offset_b = 6

TRAIN_OMEGA = True
DEFAULT_RF_OMEGA = 10.

DEFAULT_RF_ADAPTIVE_OMEGA_a = 10
DEFAULT_RF_ADAPTIVE_OMEGA_b = 50

DEFAULT_RF_THETA = 1  # .99

# Reset: Keep (1 - Zeta) of the membrane potential
# start with constant initialization
TRAIN_ZETA = False
DEFAULT_RF_ZETA = .00

DEFAULT_RF_ADAPTIVE_ZETA_a = 0
DEFAULT_RF_ADAPTIVE_ZETA_b = 0

TRAIN_DT = False
DEFAULT_DT = 0.01
DEFAULT_RF_ADAPTIVE_DT = 0.01

def sustain_osc(omega: torch.Tensor, dt: float = DEFAULT_DT) -> torch.Tensor:
    return (-1 + torch.sqrt(1 - torch.square(dt * omega))) / dt

@torch.jit.script
def step(x: torch.Tensor) -> torch.Tensor:
    #
    # x.gt(0.0).float()
    # is slightly faster (but less readable) than
    # torch.where(x > 0.0, 1.0, 0.0)
    #
    return x.gt(0.0).float()


@torch.jit.script
def gaussian(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    return (1 / (sigma * torch.sqrt(2 * torch.tensor(math.pi)))) * torch.exp(
        -((x - mu) ** 2) / (2.0 * (sigma ** 2))
    )


class StepDoubleGaussianGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors

        p = 0.15
        scale = 6.
        len = 0.5

        sigma1 = len
        sigma2 = scale * len

        gamma = 0.5
        dfd = (1. + p) * gaussian(x, mu=0., sigma=sigma1) - 2. * p * gaussian(x, mu=0., sigma=sigma2)

        return grad_output * dfd * gamma

@torch.jit.script
def FGI_DGaussian(x: torch.Tensor) -> torch.Tensor:
    x_detached = step(x).detach()

    p = 0.15
    scale = 6.
    len = 0.5

    sigma1 = len
    sigma2 = scale * len

    gamma = 0.5

    df = (1. + p) * gaussian(x, mu=0., sigma=sigma1) - 2. * p * gaussian(x, mu=0., sigma=sigma2)

    df_detached = df.detach()

    # detach of df prevents the gradients to flow through x of the gaussian function.
    dfd = gamma * df_detached * x

    dfd_detached = dfd.detach()

    return dfd - dfd_detached + x_detached

@torch.jit.script
def rf_update(
        x: torch.Tensor,  # injected current: input x weight
        u: torch.Tensor,  # membrane potential (real part)
        v: torch.Tensor,  # membrane potential (complex part)
        b: torch.Tensor,  # attraction to resting state
        omega: torch.Tensor,  # eigen ang. frequency of the neuron
        dt: float = DEFAULT_DT,  # 0.01
        theta: float = DEFAULT_RF_THETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # # membrane update (complex)
    # u = u + u.mul(torch.complex(b, omega)).mul(dt) + x.mul(dt)
    u_ = u + b * u * dt - omega * v * dt + x * dt
    v = v + omega * u * dt + b * v * dt

    # generate spike
    # z = functional.StepDoubleGaussianGrad.apply(u.real - theta)
    z = FGI_DGaussian(u_ - theta)

    # no reset or
    # soft reset # hard reset
    # u_ = u_ - z * theta  # * u_
    # v = v - z * theta # * v

    return z, u_, v

@torch.jit.script
def hrf_update(
        x: torch.Tensor,  # injected current: input x weight
        u: torch.Tensor,  # membrane potential (complex value)
        v: torch.Tensor,
        ref_period: torch.Tensor,
        b: torch.Tensor,  # attraction to resting state
        omega: torch.Tensor,  # eigen ang. frequency of the neuron
        dt: float = DEFAULT_DT,  # torch.Tensor 0.01
        theta: float = DEFAULT_RF_THETA,
):
    # damped oscillatory activity dim = (1, hidden_size)
    # membrane update u dim = (batch_size, hidden_size)
    v = v + u.mul(dt)
    u = u + x.mul(dt) - b.mul(u).mul(2 * dt) - torch.square(omega).mul(v).mul(dt)

    # generate spike
    z = StepDoubleGaussianGrad.apply(u - theta - ref_period)
    ref_period = ref_period.mul(0.9) + z

    # reset membrane potential
    # u = u.mul(1 - z.mul(theta).mul(zeta))
    # v = v.mul(1 - z.mul(theta).mul(zeta))
    return z, u, v, ref_period


################################################################
# Layer classes
################################################################
class HRFCell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            layer_size: int,
            mask_prob: float = DEFAULT_MASK_PROB,
            b_offset: float = DEFAULT_RF_B_offset,
            adaptive_b_offset: bool = TRAIN_B_offset,
            adaptive_b_offset_a: float = DEFAULT_RF_ADAPTIVE_B_offset_a,
            adaptive_b_offset_b: float = DEFAULT_RF_ADAPTIVE_B_offset_b,
            omega: float = DEFAULT_RF_OMEGA,
            adaptive_omega: bool = TRAIN_OMEGA,
            adaptive_omega_a: float = DEFAULT_RF_ADAPTIVE_OMEGA_a,
            adaptive_omega_b: float = DEFAULT_RF_ADAPTIVE_OMEGA_b,
            dt: float = DEFAULT_DT,
            bias: bool = False
    ) -> None:
        super(HRFCell, self).__init__()

        self.input_size = input_size
        self.layer_size = layer_size

        # LinearMask: applies mask only to hidden recurrent weights in forward pass
        # linear.weight initialized with xavier_uniform_
        # self.mask_prob = mask_prob
        #
        # self.linear = rf.LinearMask(
        #     in_features=input_size,
        #     out_features=layer_size,
        #     bias=bias,
        #     mask_prob=mask_prob,
        #     lbd=input_size - layer_size,
        #     ubd=input_size,
        # )

        self.linear = torch.nn.Linear(
            in_features=input_size,
            out_features=layer_size,
            bias=bias
        )

        torch.nn.init.xavier_uniform_(self.linear.weight)

        self.adaptive_omega = adaptive_omega
        self.adaptive_omega_a = adaptive_omega_a
        self.adaptive_omega_b = adaptive_omega_b

        omega = omega * torch.ones(layer_size)

        if adaptive_omega:
            self.omega = torch.nn.Parameter(omega)
            torch.nn.init.uniform_(self.omega, adaptive_omega_a, adaptive_omega_b)
        else:
            self.register_buffer('omega', omega)

        self.adaptive_b_offset = adaptive_b_offset
        self.adaptive_b_a = adaptive_b_offset_a
        self.adaptive_b_b = adaptive_b_offset_b

        b_offset = b_offset * torch.ones(layer_size)

        if adaptive_b_offset:
            self.b_offset = torch.nn.Parameter(b_offset)
            torch.nn.init.uniform_(self.b_offset, adaptive_b_offset_a, adaptive_b_offset_b)
        else:
            self.register_buffer('b_offset', b_offset)

        self.dt = dt

    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        z, u, v, ref_period = state

        in_sum = self.linear(x)

        omega = torch.abs(self.omega)

        b_offset = torch.abs(self.b_offset)

        b = omega.square().mul(0.005) + b_offset + ref_period

        z, u, v, ref_period = hrf_update(
            x=in_sum,
            u=u,
            v=v,
            ref_period=ref_period,
            b=b,
            omega=omega,
            dt=self.dt,
        )

        return z, u, v, ref_period

@torch.jit.script
def brf_update(
        x: torch.Tensor,  # injected current: input x weight
        u: torch.Tensor,  # membrane potential (real part)
        v: torch.Tensor,  # membrane potential (complex part)
        q: torch.Tensor,  # refractory period
        b: torch.Tensor,  # attraction to resting state
        omega: torch.Tensor,  # eigen ang. frequency of the neuron
        dt: float = DEFAULT_DT,  # 0.01
        theta: float = DEFAULT_RF_THETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # membrane update u dim = (batch_size, hidden_size)
    # u = u + u.mul(torch.complex(b, omega)).mul(dt) + x.mul(dt)
    u_ = u + b * u * dt - omega * v * dt + x * dt
    v = v + omega * u * dt + b * v * dt

    # # generate spike
    # z = functional.FGI_DGaussian(u_ - theta - q)
    z = StepDoubleGaussianGrad.apply(u_ - theta - q)

    q = q * 0.9 + z

    return z, u_, v, q


class RFCell(torch.nn.Module):
    def __init__(
            self,
            layer_size: int,
            mask_prob: float = DEFAULT_MASK_PROB,
            b_offset: float = DEFAULT_RF_B_offset,
            adaptive_b_offset: bool = TRAIN_B_offset,
            adaptive_b_offset_a: float = DEFAULT_RF_ADAPTIVE_B_offset_a,
            adaptive_b_offset_b: float = DEFAULT_RF_ADAPTIVE_B_offset_b,
            omega: float = DEFAULT_RF_OMEGA,
            adaptive_omega: bool = TRAIN_OMEGA,
            adaptive_omega_a: float = DEFAULT_RF_ADAPTIVE_OMEGA_a,
            adaptive_omega_b: float = DEFAULT_RF_ADAPTIVE_OMEGA_b,
            dt: float = DEFAULT_DT,
    ) -> None:
        super(RFCell, self).__init__()
        self.layer_size = layer_size

        self.adaptive_omega = adaptive_omega
        self.adaptive_omega_a = adaptive_omega_a
        self.adaptive_omega_b = adaptive_omega_b

        omega = omega * torch.ones(layer_size)

        if adaptive_omega:
            self.omega = torch.nn.Parameter(omega)
            torch.nn.init.uniform_(self.omega, adaptive_omega_a, adaptive_omega_b)
        else:
            self.register_buffer('omega', omega)

        self.adaptive_b_offset = adaptive_b_offset
        self.adaptive_b_a = adaptive_b_offset_a
        self.adaptive_b_b = adaptive_b_offset_b

        b_offset = b_offset * torch.ones(layer_size)

        if adaptive_b_offset:
            self.b_offset = torch.nn.Parameter(b_offset)
            torch.nn.init.uniform_(self.b_offset, adaptive_b_offset_a, adaptive_b_offset_b)
        else:
            self.register_buffer('b_offset', b_offset)

        self.dt = dt

    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # in_sum = self.linear(x)

        z, u, v = state

        omega = torch.abs(self.omega)

        b = -torch.abs(self.b_offset)

        z, u, v = rf_update(
            x=x,
            u=u,
            v=v,
            b=b,
            omega=omega,
            dt=self.dt,
        )

        return z, u, v


class BRFCell(RFCell):
    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # in_sum = self.linear(x)

        z, u, v, q = state

        omega = torch.abs(self.omega)

        p_omega = sustain_osc(omega)

        b_offset = torch.abs(self.b_offset)

        # divergence boundary
        b = p_omega - b_offset - q

        z, u, v, q = brf_update(
            x=x,
            u=u,
            v=v,
            q=q,
            b=b,
            omega=omega,
            dt=self.dt,
        )

        return z, u, v, q


class BHRFCell(BRFCell):
    def forward(self, x: torch.Tensor, state=None):
        if state is None:
            zero_state = torch.zeros_like(x[0])
            state = (zero_state, zero_state, zero_state, zero_state)
        elif isinstance(state, list):
            state = tuple(state)

        # Ensure state tensors are on the same device and have the same shape as x[t]
        if any(not isinstance(s, torch.Tensor) for s in state):
            zero_state = torch.zeros_like(x[0])
            state = tuple(
                zero_state if not isinstance(s, torch.Tensor) else s for s in state
            )

        T = x.shape[0]
        s_list = []
        for t in range(T):
            z, u, v, q = super().forward(x[t], state)
            state = (z, u, v, q)
            s_list.append(z)

        return torch.stack(s_list)
