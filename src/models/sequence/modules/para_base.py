# -*- coding: utf-8 -*-
"""
Created on August 2024

@author: Arnaud Yarga
"""

import torch
import numpy as np


class Base(torch.nn.Module):
    """
    Base class for creating a spiking neural layer using PyTorch.

    Parameters:
    - n_neuron (int): Number of neurons in the layer.
    - recurrent (bool): Flag to determine if the neurons should be recurrent.
    - fire (bool): Flag to determine if the neurons should fire spikes.
    - recurrent_fire (bool): Flag to determine if recurrent neurons should fire.
    - spk_threshold (float or list): Spiking threshold(s) for the neurons.
                                     Can be a single value or a list with length equal to n_neuron.
    - learn_threshold (bool): Whether the spiking threshold is learnable.
    - tau_mem (float or list): Membrane time constant. Can be a single value or a list of values.
    - tau_syn (float or list): Synaptic time constant. Can be a single value or a list of values.
    - time_step (float): Step size for updating the neuron model.
    - learn_tau (bool): Whether the time constants are learnable.
    - device (torch.device): Device to use for tensor computations, such as 'cpu' or 'cuda'.
    - debug (bool): Flag to turn on/off debugging mode.
    """

    def __init__(self, n_neuron, recurrent, fire, recurrent_fire, spk_threshold, learn_threshold, tau_mem, tau_syn,
                 time_step, learn_tau, device, debug):

        super(Base, self).__init__()
        # Validate input sizes if lists are provided for spk_threshold, tau_mem, or tau_syn
        if type(spk_threshold) == list:
            assert len(
                spk_threshold) == n_neuron, f"'spk_threshold' size ({len(spk_threshold)}) should be the same as 'n_neuron' ({n_neuron})"
        if type(tau_mem) == list:
            assert len(
                tau_mem) == n_neuron, f"'tau_mem' size ({len(tau_mem)}) should be the same as 'n_neuron' ({n_neuron})"
        if type(tau_syn) == list:
            assert len(
                tau_syn) == n_neuron, f"'tau_syn' size ({len(tau_syn)}) should be the same as 'n_neuron' ({n_neuron})"

        self.n_neuron = n_neuron
        self.recurrent = recurrent
        self.fire = fire
        self.recurrent_fire = recurrent_fire
        self.learn_threshold = learn_threshold
        self.learn_tau = learn_tau
        self.device = device
        self.debug = debug
        self.register_buffer('nb_spike_per_neuron', torch.zeros(self.n_neuron, device=self.device))

        # Initialize spiking threshold parameter
        if self.learn_threshold:
            self._spk_threshold = torch.nn.Parameter(torch.tensor(spk_threshold, device=self.device))
        else:
            self.register_buffer('_spk_threshold', torch.tensor(spk_threshold, device=self.device))

        # Neuron time constants for synaptic and membrane potentials
        alpha = torch.exp(
            -time_step / torch.tensor(tau_syn, device=self.device)) if tau_syn is not None else torch.tensor(0.,
                                                                                                             device=self.device)
        beta = torch.exp(-time_step / torch.tensor(tau_mem, device=self.device))
        if self.learn_tau:
            if tau_syn is not None:
                self._alpha = torch.nn.Parameter(alpha)
            else:
                self.register_buffer('_alpha', alpha)  # if tau_syn is None its not learned
            self._beta = torch.nn.Parameter(beta)
        else:
            self.register_buffer('_alpha', alpha)
            self.register_buffer('_beta', beta)

        # Fully connected layer for recurrent synapses
        if self.recurrent:
            self.fc_recu = torch.nn.Linear(self.n_neuron, self.n_neuron, device=self.device)
            # Initialize weights for the recurrent layer
            torch.nn.init.kaiming_uniform_(self.fc_recu.weight, a=0, mode='fan_in', nonlinearity='linear')
            torch.nn.init.zeros_(self.fc_recu.bias)
            if self.debug:
                torch.nn.init.ones_(self.fc_recu.weight)  # Optionally initialize to ones for debugging

    # Clamp the spk_threshold when learning to avoid negative thresholds
    @property
    def spk_threshold(self):
        return self._spk_threshold.relu() if self.learn_threshold else self._spk_threshold

    @spk_threshold.setter
    def spk_threshold(self, val):
        if isinstance(self._spk_threshold, torch.nn.Parameter):
            self._spk_threshold.data = val
        else:
            self._spk_threshold = val

    # Apply sigmoid as a clamp function to keep alpha in [0, 1] range as suggested in PLIF
    #  (https://arxiv.org/abs/2007.05785)
    @property
    def alpha(self):
        # if tau_syn is None, self._alpha will not be a parameter then alpha will stay zero
        return self._alpha.sigmoid() if (
                    self.learn_tau and isinstance(self._alpha, torch.nn.Parameter)) else self._alpha

    @alpha.setter
    def alpha(self, val):
        if isinstance(self._alpha, torch.nn.Parameter):
            self._alpha.data = val
        else:
            self._alpha = val

    # Apply sigmoid as a clamp function to keep beta in [0, 1] range as suggested in PLIF
    #  (https://arxiv.org/abs/2007.05785)
    @property
    def beta(self):
        return self._beta.sigmoid() if self.learn_tau else self._beta

    @beta.setter
    def beta(self, val):
        if isinstance(self._beta, torch.nn.Parameter):
            self._beta.data = val
        else:
            self._beta = val

    def extra_repr(self):
        """
        Provides additional representation for the module, useful for printing.
        """
        return f"n_neuron={self.n_neuron}, recurrent={self.recurrent}, fire={self.fire}, recurrent_fire={self.recurrent_fire}, learn_threshold={self.learn_threshold}, learn_tau={self.learn_tau}"


# Fast Sigmoid Surrogate gradient implementation from https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb
class SurrGradSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input)
        ctx.scale = scale
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.scale * torch.abs(input) + 1.0) ** 2
        return grad, None


# ATan surrogate gradient from https://proceedings.neurips.cc/paper/2021/hash/afe434653a898da20044041262b3ac74-Abstract.html
class AtanSurrGradSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input)
        ctx.scale = scale
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.scale
        grad = alpha / 2 / (1 + (np.pi / 2 * alpha * x).pow_(2)) * grad_output
        return grad, None


class SurrogateFunction:
    def __init__(self, mode="fastsig", scale=None):
        """
        Surrogate function selector class to handle different types of surrogate gradients.

        Parameters:
        - mode (str): Type of surrogate gradient function ('fastsig' or 'atan').
        - scale (float): Scaling factor for the surrogate gradient function. Default depends on mode.
        """
        super(SurrogateFunction, self).__init__()
        self.mode = mode
        if self.mode == "atan":
            self.surrogate_fn = AtanSurrGradSpike
            self.scale = 2.0
        else:
            self.surrogate_fn = SurrGradSpike
            self.scale = 100.0
        if scale:
            self.scale = scale

    def __call__(self, inputs):
        """
        Apply the surrogate gradient function to the inputs.

        Parameters:
        - inputs (torch.Tensor): The input tensor on which to apply the surrogate gradient function.

        Returns:
        - torch.Tensor: The output after applying the surrogate gradient.
        """
        return self.surrogate_fn.apply(inputs, self.scale)





class ParaLIF(Base):
    """
    Class for implementing a Parallelizable Leaky Integrate-and-Fire (ParaLIF) neuron model

    Parameters:
    - n_neuron (int): number of neurons in the layer.
    - spike_mode (str): "GS", "SB", "TRB", "D", "SD", "TD", "TRD", "T", "ST", "TT" or "TRT".
    - recurrent (bool, optional): flag to determine if the neurons should be recurrent (default: False).
    - fire (bool, optional): flag to determine if the neurons should fire spikes or not (default: True).
    - recurrent_fire (bool): Flag to determine if recurrent neurons should fire (default: True).
    - spk_threshold (float or list): Spiking threshold(s) for the neurons. Can
                    be a single value or a list with length equal to n_neuron (default: 1.).
    - learn_threshold (bool): Whether the spiking threshold is learnable (default: False).
    - tau_mem (float or list): Membrane time constant. Can be a single value or a list of values (default: 1e-3).
    - tau_syn (float or list): Synaptic time constant. Can be a single value or a list of values (default: 1e-3).
    - time_step (float): Step size for updating the neuron model (default: 1e-3).
    - learn_tau (bool): Whether the time constants are learnable (default: False).
    - device (torch.device): Device to use for tensor computations, such as 'cpu' or 'cuda'.
    - surrogate_mode (str): Type of surrogate gradient function ('fastsig' or 'atan') (default: 'fastsig').
    - surrogate_scale (float): Scaling factor for the surrogate gradient function. Default depends on mode.
    - debug (bool, optional): flag to turn on/off debugging mode (default: False).
    """

    def __init__(self, n_neuron, spike_mode, recurrent=False, fire=True, recurrent_fire=True, spk_threshold=1.,
                 learn_threshold=False, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, learn_tau=False,
                 device=None, surrogate_mode="fastsig", surrogate_scale=None, debug=False):
        """
        Initializes the ParaLIF model with the specified parameters.
        """

        # Call the constructor of the Base class
        super(ParaLIF, self).__init__(n_neuron, recurrent, fire, recurrent_fire, spk_threshold, learn_threshold,
                                      tau_mem, tau_syn, time_step, learn_tau, device, debug)

        # Set the spiking function based on the spike_mode and other parameters
        self.spike_mode = spike_mode
        self.spike_fn = SpikingFunction(self.spike_mode, surrogate_mode, surrogate_scale, self.device) if (
                    self.fire or (self.recurrent and self.recurrent_fire)) else None

        # Initialize the number of spikes per neuron for recording
        self.register_buffer('nb_spike_per_neuron_rec', torch.zeros(self.n_neuron, device=self.device))
        self.nb_steps = None

    def compute_params_fft(self, nb_steps):
        """
        Compute the FFT of the leakage parameters for parallel Leaky Integration.

        Returns:
        - fft_l_k: Product of FFT of parameters l and k
        """
        if nb_steps is None: return None
        self.device = self.alpha.device
        # Compute FFT for alpha parameter
        if self.alpha.dim() == 0:
            l = torch.pow(self.alpha, torch.arange(nb_steps, device=self.device))
            fft_l = torch.fft.rfft(l, n=2 * nb_steps).unsqueeze(1)
        elif self.alpha.dim() > 0:
            l = torch.pow(self.alpha, torch.arange(nb_steps, device=self.device).expand(self.n_neuron, nb_steps).T)
            fft_l = torch.fft.rfft(l, n=2 * nb_steps, dim=0)

        # Compute FFT for beta parameter
        if self.beta.dim() == 0:
            k = torch.pow(self.beta, torch.arange(nb_steps, device=self.device)) * (1 - self.beta)
            fft_k = torch.fft.rfft(k, n=2 * nb_steps).unsqueeze(1)
        elif self.beta.dim() > 0:
            k = torch.pow(self.beta, torch.arange(nb_steps, device=self.device).expand(self.n_neuron, nb_steps).T) * (
                        1 - self.beta)
            fft_k = torch.fft.rfft(k, n=2 * nb_steps, dim=0)

        # Return the product of FFTs of l and k
        return fft_l * fft_k

    def forward(self, X, parallel=True):
        X = X.permute(1, 0, 2).contiguous() # new add
        """
        Perform forward pass of the network.

        Parameters:
        - X (tensor): Input tensor with shape (batch_size, nb_steps, input_size)
        - parallel (bool, optional): If 'True' (default) the parallel forward is used, 
          and if 'False' the sequential forward is used

        Returns:
        - Return membrane potential tensor with shape (batch_size, nb_steps, n_neuron) if 'fire' is False
        - Return spiking tensor with shape (batch_size, nb_steps, n_neuron) if 'fire' is True
        - Return the tuple (spiking tensor, membrane potential tensor) if 'debug' is True and 'fire' is True
        """

        # If parallel is False, use the sequential forward method
        if not parallel: return self.forward_sequential(X)

        batch_size, nb_steps, _ = X.shape

        # Compute FFT parameters if the number of steps has changed or if tau params are learnable
        if self.nb_steps != nb_steps or self.learn_tau:
            self.fft_l_k = self.compute_params_fft(nb_steps)
            self.nb_steps = nb_steps

        # Perform parallel leaky integration - Equation (15)
        fft_X = torch.fft.rfft(X, n=2 * nb_steps, dim=1)
        mem_pot_hidden = torch.fft.irfft(fft_X * self.fft_l_k, n=2 * nb_steps, dim=1)[:, :nb_steps, :]

        # Handle recurrent connections if needed
        if self.recurrent:
            mem_pot_hidden_ = F.pad(mem_pot_hidden, (0, 0, 1, 0), "constant", 0)[:, :-1]
            # Compute hidden state - Equation (22)
            hidden_state = self.spike_fn(mem_pot_hidden_, self.spk_threshold) if self.recurrent_fire else F.relu(
                mem_pot_hidden_)
            if self.recurrent_fire: self.nb_spike_per_neuron_rec = torch.mean(torch.mean(hidden_state, dim=0), dim=0)
            # Perform parallel leaky integration for input and hidden state - Equation (23)
            fft_X_hidden_state = torch.fft.rfft(X + self.fc_recu(hidden_state), n=2 * nb_steps, dim=1)
            mem_pot_temp = torch.fft.irfft(fft_X_hidden_state * self.fft_l_k, n=2 * nb_steps, dim=1)[:, :nb_steps, :]
            mem_pot_final = mem_pot_hidden + mem_pot_temp
        else:
            mem_pot_final = mem_pot_hidden

        # Handle firing if enabled
        if self.fire:
            # Perform firing - Equation (24)
            spikes = self.spike_fn(mem_pot_final, self.spk_threshold)
            self.nb_spike_per_neuron = torch.mean(torch.mean(spikes, dim=0), dim=0)
            spikes = spikes.permute(1, 0, 2).contiguous() #new add
            return (spikes, mem_pot_final) if self.debug else spikes

        return mem_pot_final

    def forward_sequential(self, X):
        """
        Sequential implementation of the ParaLIF forward function.
        """
        batch_size, nb_steps, _ = X.shape
        syn_cur_hidden = torch.zeros_like(X[:, 0])  # Initialize synaptic current for hidden layer
        mem_pot_hidden = torch.zeros_like(X[:, 0])  # Initialize membrane potential for hidden layer
        mem_pot_hidden_prev = torch.zeros_like(X[:, 0])  # Initialize previous membrane potential

        if self.recurrent:
            syn_cur_temp = torch.zeros_like(X[:, 0])  # Initialize synaptic current for recurrent connections
            mem_pot_temp = torch.zeros_like(X[:, 0])  # Initialize membrane potential for recurrent connections
            hidden_state = torch.zeros_like(X[:, 0])  # Initialize hidden state tensor

        # mem_pot_final = torch.zeros_like(X) # Initialize final membrane potential
        mem_pot_final = []
        # spikes = torch.zeros_like(X) # Initialize spikes tensor
        spikes_t = torch.zeros_like(X[:, 0])
        spikes = []

        for t in range(nb_steps):
            # Integrate input to synaptic current
            syn_cur_hidden = self.alpha * syn_cur_hidden + X[:, t]
            mem_pot_hidden_prev = mem_pot_hidden

            # Integrate synaptic current to membrane potential - Equation (7)
            mem_pot_hidden = self.beta * mem_pot_hidden_prev + (1 - self.beta) * syn_cur_hidden

            if self.recurrent:
                # Integrate input and hidden state to recurrent synaptic current
                syn_cur_temp = self.alpha * syn_cur_temp + X[:, t] + self.fc_recu(hidden_state)

                # Integrate recurrent synaptic current to recurrent membrane potential
                mem_pot_temp = self.beta * mem_pot_temp + (1 - self.beta) * syn_cur_temp
                # mem_pot_final[:, t] = mem_pot_hidden + mem_pot_temp
                mem_pot_final.append(mem_pot_hidden + mem_pot_temp)

                # Compute hidden state - Equation (22)
                hidden_state = self.spike_fn(torch.stack((mem_pot_hidden_prev, mem_pot_hidden), dim=1),
                                             self.spk_threshold)[:, -1] if self.recurrent_fire else F.relu(
                    mem_pot_hidden)
            else:
                # mem_pot_final[:, t] = mem_pot_hidden
                mem_pot_final.append(mem_pot_hidden)

            # Handle firing
            if self.fire:
                # spikes[:,t] = self.spike_fn(mem_pot_final[:,[t-1,t]], self.spk_threshold)[:,-1]
                mem_pot_final_prev = mem_pot_final[-2] if len(mem_pot_final) > 1 else torch.zeros_like(
                    mem_pot_final[-1])
                spikes_t = self.spike_fn(torch.stack((mem_pot_final_prev, mem_pot_final[-1]), dim=1),
                                         self.spk_threshold)[:, -1]
            spikes.append(spikes_t)

        mem_pot_final = torch.stack(mem_pot_final, dim=1)
        spikes = torch.stack(spikes, dim=1)
        # Save average spikes and return ouputs tensors
        if self.fire:
            self.nb_spike_per_neuron = torch.mean(torch.mean(spikes, dim=0), dim=0)
            if self.recurrent_fire: self.nb_spike_per_neuron_rec = torch.mean(torch.mean(spikes, dim=0), dim=0)
            return (spikes, mem_pot_final) if self.debug else spikes

        return mem_pot_final

    def extra_repr(self):
        """
        Provides additional representation for the module, useful for printing.
        """
        return f"spike_mode={self.spike_mode}, " + super().extra_repr()


def identity_fn(x):
    return x


def tanh_relu_fn(x):
    return F.relu(torch.tanh(x))


class SpikingFunction(torch.nn.Module):
    """
    Perform spike generation using various methods. The main spiking methods include:
        - GS : Gumbel Softmax
        - SB : Sigmoid Bernoulli
        - D : Delta
        - T : Threshold
    Variants of these methods can be applied by first normalizing the input using Sigmoid or Hyperbolic Tangent functions.
    """

    def __init__(self, spike_mode, surrogate_mode, surrogate_scale, device):
        """
        Initializes the SpikingFunction with the specified spike generation mode and other parameters.

        Parameters:
        - spike_mode (str): The spiking method to use. Must be one of the available modes in `spike_mode_list`.
        - surrogate_mode (str): The surrogate gradient mode used in some spiking methods.
        - surrogate_scale (float): Scale parameter for the surrogate gradient.
        - device (torch.device): Device to use for computations, such as 'cpu' or 'cuda'.
        """
        super(SpikingFunction, self).__init__()
        self.spike_mode_list = ["GS", "SB", "TRB", "D", "SD", "TD", "TRD", "T", "ST", "TT", "TRT"]
        assert spike_mode in self.spike_mode_list, f"'{spike_mode}' spike mode is not available. The available options are {self.spike_mode_list}"
        self.spike_mode = spike_mode

        # Input normalization function based on the selected spiking mode
        if spike_mode in ["SB", "SD", "ST"]:
            self.normalize = torch.sigmoid  # Sigmoid normalization for certain spike modes
        elif spike_mode in ["TD", "TT"]:
            self.normalize = torch.tanh  # Tanh normalization for certain spike modes
        elif spike_mode in ["TRB", "TRD", "TRT"]:
            self.normalize = tanh_relu_fn  # Combination of Tanh and ReLU for specific modes
        else:
            self.normalize = identity_fn  # No normalization for other modes

        # Spike generation function selection based on the spike_mode
        if spike_mode in ["SB", "TRB"]:
            self.generate = self.bernoulli_fn  # Use Bernoulli spike generation
        elif spike_mode in ["D", "SD", "TD", "TRD"]:
            self.surrogate_fn = SurrogateFunction(mode=surrogate_mode, scale=surrogate_scale)
            self.generate = self.delta_fn  # Use Delta spike generation
        elif spike_mode in ["T", "ST", "TT", "TRT"]:
            self.surrogate_fn = SurrogateFunction(mode=surrogate_mode, scale=surrogate_scale)
            self.generate = self.threshold_fn  # Use Threshold spike generation
        elif spike_mode == "GS":
            raise NotImplementedError
            # self.gs = GumbelSoftmax(device)
            # self.generate = self.gumbel_softmax_fn  # Use Gumbel Softmax spike generation

    def forward(self, inputs, spk_threshold):
        """
        Perform spike generation on the input tensor based on the selected spiking method.

        Parameters:
        - inputs (tensor): The input tensor to be processed.
        - spk_threshold (float): The threshold value for spike generation.

        Returns:
        - tensor: The output tensor representing generated spikes.
        """
        inputs = self.normalize(inputs)  # Normalize the input
        return self.generate(inputs, spk_threshold)  # Generate spikes based on the selected method

    # Delta Spikes Generation - Uses surrogate gradient
    def delta_fn(self, inputs, spk_threshold):
        """
        Generate spikes using the Delta method with a surrogate gradient.

        Parameters:
        - inputs (tensor): The input tensor with shape (batch_size, time_steps, input_size).
        - spk_threshold (float): The threshold for spike generation.

        Returns:
        - tensor: The generated spike tensor.
        """
        inputs_previous = F.pad(inputs, (0, 0, 1, 0), "constant", 0)[:, :-1]  # Previous time step inputs
        return self.surrogate_fn((inputs - inputs_previous) - spk_threshold)  # Spike based on change from previous step

    # Threshold Spikes Generation - Simple threshold comparison
    def threshold_fn(self, inputs, spk_threshold):
        """
        Generate spikes using a simple threshold comparison.

        Parameters:
        - inputs (tensor): The input tensor.
        - spk_threshold (float): The threshold for spike generation.

        Returns:
        - tensor: The generated spike tensor.
        """
        return self.surrogate_fn(inputs - spk_threshold)

    def extra_repr(self):
        """
        Provides additional representation for the module, useful for printing.
        """
        desc = {"D": "Delta", "T": "Threshold", "B": "Bernoulli", "S": "Gumbel Softmax"}
        spike_mode_text = ""
        if self.spike_mode in ["SB", "SD", "ST"]:
            spike_mode_text = "Sigmoid "
        elif self.spike_mode in ["TD", "TT"]:
            spike_mode_text = "Tanh "
        elif self.spike_mode in ["TRB", "TRD", "TRT"]:
            spike_mode_text = "Tanh-ReLU "
        spike_mode_text += desc[self.spike_mode[-1]]
        return f"spike_mode: {spike_mode_text}"
