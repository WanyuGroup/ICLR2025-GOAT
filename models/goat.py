import numpy as np
import math
import torch
from torch.nn import functional as F
from utils import utilis_func as uf
import os
from models.egnn_blocks import EGNN_dynamics
from torchdiffeq import odeint
from copy import deepcopy
from models.support_model import icp
from scipy.optimize import linear_sum_assignment

def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

def inv_cdf(t):  # this is used to reweight the property.
    return 1 - torch.sqrt(1 - t)


def inv_sin(t):
    return 1 - torch.sin(torch.pi * t / 2)


def T(t):
    # 0   0, 1 beta_max
    beta_min = 0.1
    beta_max = 20
    return 0.5 * (beta_max - beta_min) * t**2 + beta_min * t


def T_hat(t):
    # 0 beta_min, 1 beta_max
    beta_min = 0.1
    beta_max = 20
    return (beta_max - beta_min) * t + beta_min

def polynomial_schedule_(t, s=1e-7, power=2.0):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    # steps = timesteps + 1
    # x = np.linspace(0, steps, steps)
    alphas2 = (1 - t**power) ** 2
    # alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
    precision = 1 - 2 * s
    alphas2 = precision * alphas2 + s  # numerical stability.
    return alphas2

def VP_path(x, t):
    # t in zeros and ones
    # if noi
    beta_min = 0.1
    beta_max = 20
    # u = 1 - t
    # t = 1 - t # Reverse time, x0 for sample, x1 for noise
    log_mean_coeff = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    # log_mean_coeff.to(x.device)
    mean = torch.exp(log_mean_coeff[:, None, None]) * x
    std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
    return mean, std

def VP_field(x0, xt, t):
    M_para = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)  # add epsilon to stable it
    M_para = M_para[:, None, None]
    vector = (
        torch.exp(-T(t))[:, None, None] * xt
        - torch.exp(-0.5 * T(t))[:, None, None] * x0
    )

    return -vector

# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)

def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

def loss_reduce_mean_except_batch_with_mask(loss, mask):
    """
    Args:
        loss: [b, n, 3]
        mask: [b, n, 1]
    """
    if len(loss.shape) == 3:
        losses = loss.sum(-1)  # [b, n]
    else:
        losses = loss
    if len(mask.shape) == 3:
        mask = mask.squeeze(-1)
    return (losses * mask).sum(-1) / mask.sum(-1)


def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(
        zeros + 0.5 * torch.log(2 * np.pi * sigma ** 2) + 0.5
    )


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
        (
                torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8)
                + 0.5 * (q_sigma ** 2 + (q_mu - p_mu) ** 2) / (p_sigma ** 2)
                - 0.5
        ) * node_mask
    )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu) ** 2)
    assert len(q_sigma.size()) == 1 or len(q_sigma.size()) == 0
    assert len(p_sigma.size()) == 1 or len(p_sigma.size()) == 0
    return (d * torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8)
            + 0.5 * (d * q_sigma ** 2 + mu_norm2) / (p_sigma ** 2)
            - 0.5 * d
            )


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""

    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


class GeometricOptimalTransportFlow(torch.nn.Module):
    """
    The Geometric Optimal Transport Flow (GOAT).
    """

    def __init__(
            self,
            dynamics: EGNN_dynamics,
            in_node_nf: int,
            n_dims: int,
            timesteps: int = 10000,
            parametrization="eps",
            time_embed=False,
            vae=None,
            loss_type="ot",
            norm_values=(1.0, 1.0, 1.0),
            norm_biases=(None, 0.0, 0.0),
            include_charges=True,
            discrete_path="OT_path",
            cat_loss="l2",
            cat_loss_step=-1,
            on_hold_batch=-1,
            sampling_method="vanilla",
            weighted_methods="jump",
            ode_method="dopri5",
            without_cat_loss=False,
            angle_penalty=False,
            extend_feature_dim=0,
            minimize_type_entropy=False,
            node_classifier_model_ckpt=None,
            minimize_entropy_grad_coeff=0.5,
            trainable_ae=False,
            sigma=0,
            ot_method='exact',
            distill=False,
            device='gpu'
    ):
        super().__init__()

        # assert loss_type in {'ot'}
        self.set_odeint(method=ode_method)
        self.loss_type = loss_type
        self.include_charges = include_charges
        self._eps = 0.0  # TODO: fix the trace computation part
        self.discrete_path = discrete_path
        self.ode_method = ode_method

        self.cat_loss = cat_loss
        self.cat_loss_step = cat_loss_step
        self.on_hold_batch = on_hold_batch
        self.sampling_method = sampling_method
        self.weighted_methods = weighted_methods
        self.without_cat_loss = without_cat_loss
        self.angle_penalty = angle_penalty

        self.dynamics = dynamics

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges - extend_feature_dim
        self.extend_feature_dim = extend_feature_dim

        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.time_embed = time_embed
        self.minimize_type_entropy = minimize_type_entropy
        self.node_classifier_model_ckpt = node_classifier_model_ckpt
        self.minimize_entropy_grad_coeff = minimize_entropy_grad_coeff
        self.node_pred_model = None
        self.register_buffer("buffer", torch.zeros(1))
        self.trainable_ae = trainable_ae
        self.sigma = sigma


        if not self.trainable_ae:
            self.vae = vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            self.vae = vae.train()
            for param in self.vae.parameters():
                param.requires_grad = True
        if time_embed:
            self.register_buffer(
                "frequencies", 2 ** torch.arange(self.frequencies) * torch.pi
            )
        if self.minimize_type_entropy and not os.path.exists(
                self.node_classifier_model_ckpt
        ):
            raise ValueError(
                "node_classifier_model_ckpt must be provided if minimize_type_entropy is True"
            )
        self.distill=distill

        # if noise_schedule != 'learned':
        #     self.check_issues_norm_values()

    def set_odeint(self, method="dopri5", rtol=1e-4, atol=1e-4):
        self.method = method
        self._atol = atol
        self._rtol = rtol
        self._atol_test = 1e-7
        self._rtol_test = 1e-7

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1.0 / max_norm_value:
            raise ValueError(
                f"Value for normalization value {max_norm_value} probably too "
                f"large with sigma_0 {sigma_0:.5f} and "
                f"1 / norm_value = {1. / max_norm_value}"
            )

    def phi(self, t, x, node_mask, edge_mask, context):
        # TODO: check the frequencies buffer. input is embedding to get better performance.
        if self.time_embed:
            t = self.frequencies * t[..., None]
            t = torch.cat((t.cos(), t.sin()), dim=-1)
            t = t.expand(*x.shape[:-1], -1)

        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)

        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(
            self.norm_values[0]
        )

        # Casting to float in case h still has long or int type.
        h_cat = (
                (h["categorical"].float() - self.norm_biases[1])
                / self.norm_values[1]
                * node_mask
        )
        h_int = (h["integer"].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {"categorical": h_cat, "integer": h_int}

        return x, h, delta_log_px

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def unnormalize_z(self, z, node_mask):  # Check the unnormalize_z function
        # Parse from z
        x, h_cat = (
            z[:, :, 0: self.n_dims],
            z[:, :, self.n_dims: self.n_dims + self.num_classes],
        )
        h_int = z[
                :, :, self.n_dims + self.num_classes: self.n_dims + self.num_classes + 1
                ]

        # print("unnormalize_", h_int.size(),x.size(), h_cat.size())
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        if self.extend_feature_dim > 0:
            h_extend = z[:, :, self.n_dims + self.num_classes + 1:]
            output = torch.cat([x, h_cat, h_int, h_extend], dim=2)
        else:
            output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    # def zero_step_direction(self, xh_0,  node_mask, edge_mask, context):
    #     """Computes the direction of the zero-step flow."""
    #     zeros = torch.zeros(size=(node_mask.size(0), 1), device=node_mask.device)
    #     # gamma_0 = self.gamma(zeros)
    #     net_out = self.phi(zeros, xh_0, node_mask, edge_mask, context)

    #     return

    def sample_p_xh_given_z0(self, z0, node_mask):
        """Samples x ~ p(x|z0)."""

        # print(z0.size(),node_mask.size())
        # if self.cat_loss_step > 0:
        #     #under this case we use the direction of the network output as the categorical sampling results.
        #     predicted_0 = self.phi(0.)

        x = z0[:, :, : self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)

        # if self.include_charges:
        x, h_cat, h_int = self.unnormalize(
            x, z0[:, :, self.n_dims: self.n_dims + self.num_classes], h_int, node_mask
        )
        # else:
        #     x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:], h_int,
        #                                    node_mask)

        tensor = self.deq.reverse({"categorical": h_cat, "integer": h_int})

        one_hot, charges = tensor["categorical"], tensor["integer"]
        # h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        # h_int = torch.round(h_int).long() * node_mask
        h = {"integer": charges, "categorical": one_hot}

        return x, h

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def solve_optimal_rotation(
            self, x: np.ndarray, z: np.ndarray, node_mask: np.ndarray
    ):
        """
        x:  [b, n, 3+5]
        z:  [b, n, 3+5]
        node_mask: [b, n, 1]
        """
        ret_z = deepcopy(z)
        length = node_mask.squeeze().sum(axis=-1).astype(np.int32)  # [b]
        for _idx, l in enumerate(length):
            _, z_rotated, _ = icp(z[_idx, :l, :3], x[_idx, :l, :3])
            ret_z[_idx, :l, :3] = z_rotated
        return ret_z

    def solve_optimal_permutation(self, x: np.ndarray, z: np.ndarray, node_mask: np.ndarray):
        """
        x:  [b, n, 3+5]
        z:  [b, n, 3+5]
        node_mask: [b, n, 1]
        """
        ret_z = deepcopy(z)
        length = node_mask.squeeze().sum(axis=-1).astype(np.int32)
        distance_matrices = np.sqrt(np.sum((np.expand_dims(x, axis=2) - np.expand_dims(z, axis=1))** 2,axis=-1,))  # [b, n, n]
        for _idx, l in enumerate(length):
            _, col_ind = linear_sum_assignment(
                distance_matrices[_idx, :l, :l], maximize=False
            )
            ret_z[_idx, :l, :] = z[_idx, col_ind, :]
        return ret_z

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t
        return self.sigma

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t, xt
        return x1 - x0

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def sample_location_and_conditional_flow(self, x0, x1, node_mask, t=None, return_noise=False):
        # x0 source distribution
        # x1 target distribution (data)

        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_combined_position_feature_noise(
            x1.size(0), x1.size(1), node_mask
        )
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def solve_optimal_molecule_transport(self, _z, xh, node_mask):
        _z = self.solve_optimal_rotation(
            xh.detach().cpu().numpy(),
            _z.detach().cpu().numpy(),
            node_mask.detach().cpu().numpy(),
        )
        _z = self.solve_optimal_permutation(
            xh.detach().cpu().numpy(),
            _z,
            node_mask.detach().cpu().numpy(),
        )
        _z = torch.tensor(
            _z,
            dtype=xh.dtype,
            device=xh.device,
        )
        return _z, xh

    def optimal_transport(self, xh, node_mask):
        b, n, _ = xh.size()
        z_x = torch.randn((b, n, self.n_dims), device=node_mask.device)
        z_h = torch.randn((b, n, self.in_node_nf), device=node_mask.device)
        z = torch.cat([z_x, z_h], dim=2)
        # z, xh = self.ot_sampler.sample_plan(z, xh, node_mask)
        z = xh * node_mask
        z[:, :, :self.n_dims] = uf.remove_mean_with_mask(z[:, :, :self.n_dims], node_mask)
        return z, xh


    def compute_loss(self, x, h, node_mask, edge_mask, context, noise):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""
        b, n, _ = x.size()
        xh = torch.cat([x, h], dim=2)
        if noise is None:
            _z = self.sample_combined_position_feature_noise(b, n, node_mask)
            # _z, xh = self.optimal_transport(xh, node_mask)
        else:
            _z = noise
        _z, xh = self.solve_optimal_molecule_transport(_z, xh, node_mask)
        if self.distill:
            t = torch.zeros(xh.shape[0], device=xh.device) * (1 - 1e-3) + 1e-3
            xt = _z
            ut = xh - _z
        else:
            t, xt, ut = self.sample_location_and_conditional_flow(_z, xh, node_mask)
        vt = self.phi(t, xt, node_mask, edge_mask, context)
        loss = sum_except_batch((vt - ut).square())
        return loss, {"error": loss}

    def _gradients_from_node_type_entropy(self, x, node_mask, edge_mask):
        print(
            f"shape of x:{x.shape} shape of node_mask: {node_mask.shape}, shape of edge_mask: {edge_mask.shape}"
        )
        input_h = torch.ones(
            x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype, device=x.device
        )
        input_h = input_h / input_h.shape[-1]
        input_h = input_h.to(x.device, x.dtype)
        input_h.requires_grad = True
        xh = torch.cat([x, input_h], dim=-1)
        output = self.node_pred_model._forward(
            0, xh, node_mask, edge_mask, context=None
        )
        _h = torch.softmax(output[:, :, self.n_dims:], dim=-1)  # [B,N,K]
        print(f"===_h: {_h}")
        h_entropy = -torch.sum(_h * torch.log(_h + 1e-10), dim=-1)  # [B,N]
        print(f"===h_entropy: {h_entropy}")
        h_entropy_loss = loss_reduce_mean_except_batch_with_mask(
            h_entropy, node_mask
        ).mean()
        print(f"===h_entropy_loss: {h_entropy_loss}")
        xh_grad = torch.autograd.grad(h_entropy_loss, xh, create_graph=True)[0]
        print(f"===x_grad: {xh_grad}")

        return xh_grad[:, :, : self.n_dims]


    def decode(self, z, node_mask, edge_mask, context) -> torch.Tensor:
        self.wrapper_count = 0
        self.time_steps = []

        def wrapper(t, x):
            self.wrapper_count += 1
            self.time_steps.append(t.cpu().numpy().item())
            dx = self.phi(t, x, node_mask, edge_mask, context)
            dx = dx * node_mask
            return dx

        t_list = torch.linspace(0, 1, 2, device=z.device)
        out = odeint(
            wrapper, z, t_list, method=self.method, rtol=self._rtol, atol=self._atol
        )
        print(f"wrapper_count: {self.wrapper_count}")
        # print(f"time_steps: {self.time_steps}")
        return out

    def decode_chain(self, z, t, node_mask, edge_mask, context) -> torch.Tensor:
        # here t is all the model which we used to decode
        def wrapper(t, x):
            dx = self.phi(t, x, node_mask, edge_mask, context)
            if self.cat_loss_step > 0:
                if t > self.cat_loss_step:
                    dx[:, :, self.n_dims: -1] = 0
                else:
                    dx[:, :, self.n_dims: -1] = dx[:, :, self.n_dims: -1] / (
                        self.cat_loss_step
                    )
                # cat_mask = t.squeeze() < self.cat_loss_step
                # dx[~cat_mask][:,self.n_dims:-1] = 0
                # dx[cat_mask][:,self.n_dims:-1] = dx[cat_mask][:,self.n_dims:-1] / self.cat_loss_step # align the speed.
            if self.discrete_path == "VP_path":
                M_para = (
                        -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                )  # add epsilon to stable it
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx * M_para
            elif self.discrete_path == "HB_path":
                M_para = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims:] = dx[:, :, self.n_dims:] * M_para
            elif self.discrete_path == "VP_path_poly":
                alpha_s2 = polynomial_schedule_(t)
                M_para = 1 / (1 - alpha_s2 + 1e-5)  # alpha_div / 1 - alpha_t2
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx * M_para
            elif self.discrete_path == "HB_path_poly":
                alpha_s2 = polynomial_schedule_(t)
                M_para = 1 / (1 - alpha_s2 + 1e-5)  # alpha_div / 1 - alpha_t2
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims:] = dx[:, :, self.n_dims:] * M_para
            else:
                pass
            return dx

        t = torch.tensor(t, dtype=torch.float, device=z.device)

        return odeint(
            wrapper, z, t, method=self.method, rtol=self._rtol, atol=self._atol
        )

    def prior_likelihood(self, z, node_mask):
        z_x = z[:, :, : self.n_dims]
        z_h = z[:, :, self.n_dims:]
        # def forward(self, z_x, z_h, node_mask=None):
        assert len(z_x.size()) == 3
        assert len(node_mask.size()) == 3
        assert node_mask.size()[:2] == z_x.size()[:2]

        assert (z_x * (1 - node_mask)).sum() < 1e-8 and (
                z_h * (1 - node_mask)
        ).sum() < 1e-8, "These variables should be properly masked."

        log_pz_x = uf.center_gravity_zero_gaussian_log_likelihood_with_mask(
            z_x, node_mask
        )

        log_pz_h = uf.standard_gaussian_log_likelihood_with_mask(z_h, node_mask)

        log_pz = log_pz_x + log_pz_h

        return log_pz

    def unified_transport(self, x, h, node_mask=None, edge_mask=None, context=None, scaffold_mask=None):
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.vae.encode(x, h, node_mask, edge_mask, context)
        # Infer latent z.
        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        uf.assert_correctly_masked(z_xh_mean, node_mask)
        z_xh_sigma = torch.cat([z_x_sigma.expand(-1, -1, 3), z_h_sigma], dim=2)
        z_xh = self.vae.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        # z_xh = z_xh_mean
        z_xh = z_xh.detach()  # Always keep the encoder fixed.
        uf.assert_correctly_masked(z_xh, node_mask)

        # Compute reconstruction loss.
        if self.trainable_ae:
            xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
            # Decoder output (reconstruction).
            x_recon, h_recon = self.vae.decoder._forward(z_xh, node_mask, edge_mask, context)
            xh_rec = torch.cat([x_recon, h_recon], dim=2)
            loss_recon = self.vae.compute_reconstruction_error(xh_rec, xh)
        else:
            loss_recon = 0

        z_x = z_xh[:, :, :self.n_dims]
        z_h = z_xh[:, :, self.n_dims:]
        uf.assert_mean_zero_with_mask(z_x, node_mask)
        # Make the data structure compatible with the EnVariationalDiffusion compute_loss().
        # z_h = {'categorical': torch.zeros(0).to(z_h), 'integer': z_h}
        return z_x, z_h, loss_recon

    def compute_transport_cost(self, x, h, node_mask=None, edge_mask=None, context=None, noise=None):
        # x, h, loss_feature = self.unified_transport(x, h, node_mask, edge_mask, None)

        # Original space
        h = torch.cat([h['categorical'], h['integer']], dim=2)
        self.in_node_nf = h.size(2)

        b, n, _ = x.size()
        xh = torch.cat([x, h], dim=2)
        if noise is None:
            _z = self.sample_combined_position_feature_noise(b, n, node_mask)
        else:
            _z = noise
        _z, xh = self.solve_optimal_molecule_transport(_z, xh, node_mask)
        transport_cost = sum_except_batch((xh - _z).square())
        # transport_cost = torch.div(transport_cost, node_mask.sum(1).squeeze(-1))
        return transport_cost

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None, noise=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        x, h, loss_feature = self.unified_transport(x, h, node_mask, edge_mask, context)

        loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, noise)

        neg_log_pxh = loss

        return neg_log_pxh

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = uf.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=node_mask.device,
            node_mask=node_mask,
        )
        z_h = uf.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf),
            device=node_mask.device,
            node_mask=node_mask,
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def sample_cat_z0(self, xh, node_mask, edge_mask, context):
        """
        get the catgorical distribution according to coordinate and features.
        """
        # whether input use a xh or else.
        t = torch.zeros_like(xh[:, 0, 0]).view(-1, 1, 1)
        net_out = self.phi(0.0, xh, node_mask, edge_mask, context)
        z_h = net_out[
              :, :, self.n_dims: -1
              ]  # use the score function as the sampling direction. Instead of the ode results.
        xh[
        :, :, self.n_dims: -1
        ] = z_h  # replace the original xh with the sampled one.

        return xh

    def training_cat_z0(self, xh, node_mask, edge_mask, context):
        """
        get the categorical distribution on the zeroth term.
        """
        mask = torch.rand_like(xh[:, :, self.n_dims: -1])  # destroy signal for this.
        xh[:, :, self.n_dims: -1] = mask
        net_out = self.phi(0.0, xh, node_mask, edge_mask, context)
        # Get the categorical distribution.
        z_h = net_out[:, :, self.n_dims: -1]
        # z_h = z_h.reshape(z_h.size(0),z_h.size(1),self.n_cat,self.n_cat)
        cat_loss_zero_term = torch.nn.CrossEntropyLoss(
            z_h, xh[:, :, self.n_dims: -1].argmax(dim=2)
        )

        return cat_loss_zero_term

    def compress(self, xh, t, normalize_factor=1.0):
        """
        Compresses the time interval [0, 1] to [0, t]. used for the categorical distribution.
        """
        t = t.view(-1, 1, 1).expand_as(xh)
        t[:, :, self.n_dims: -1] = t[:, :, self.n_dims: -1] / normalize_factor
        return t

    @torch.no_grad()
    def sample(
            self,
            n_samples,
            n_nodes,
            node_mask,
            edge_mask,
            context,
            fix_noise=False,
            return_noise=False,
    ):
        """
        Draw samples from the generative model.
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(
                n_samples, n_nodes, node_mask
            )

        uf.assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)
        if self.distill:
            t = torch.zeros(z.shape[0], device=z.device) * (1 - 1e-3) + 1e-3
            dx = self.phi(t, z, node_mask, edge_mask, context)
            z_ = dx + z
        else:
            z_ = self.decode(z, node_mask, edge_mask, context)[-1]

        uf.assert_correctly_masked(z_[:,:,:self.n_dims], node_mask)
        x, h = self.vae.decode(z_, node_mask, edge_mask, context)
        if return_noise:
            return x, h, z
        else:
            return x, h

    @torch.no_grad()
    def sample_chain(
            self,
            dequantizer,
            n_samples,
            n_nodes,
            node_mask,
            edge_mask,
            context,
            keep_frames=None,
    ):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        uf.assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)
        if keep_frames is None:
            keep_frames = 100
        else:
            assert keep_frames <= 1000

        # chain = torch.zeros((keep_frames,) + z.size(), device=z.device)
        time_step = list(np.linspace(1, 0, keep_frames))

        chain_z = self.decode_chain(z, time_step, node_mask, edge_mask, context)

        for i in range(len(chain_z) - 1):
            ##fix chain sampling
            chain_z[i] = self.unnormalize_z(chain_z[i], node_mask)
            # chain_z[i] =
            # one_hot = chain_z[i][:, :, 3:8]
            # charges = chain_z[i][:, :, 8:]
            # tensor = dequantizer.reverse({'categorical': one_hot, 'integer': charges})
            # one_hot, charges = tensor['categorical'], tensor['integer']
            # chain_z[i] = torch.cat([chain_z[i][:, :, :3], one_hot, charges], dim=2)

        chain_z = reversed(chain_z)
        x, h = self.sample_p_xh_given_z0(
            dequantizer, chain_z[-1], node_mask
        )  # TODO this should be the reverse of our flow model
        uf.assert_mean_zero_with_mask(x[:, :, : self.n_dims], node_mask)
        b, n, _ = x.size()
        distance2origin = x.square().sum(dim=-1).sqrt()  # [b, n]
        if self.extend_feature_dim > 0:
            extend_feat = self.extend_feature_embedding(
                distance2origin.view(-1)
            )  # [b, n] -> [b, n, dim]
            extend_feat = extend_feat.view(b, n, -1)  # [bxn, dim] -> [b, n, dim]
            xh = torch.cat([x, h["categorical"], h["integer"], extend_feat], dim=2)
        else:
            xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)

        # print(chain_z.size(),xh.size(),h['integer'], h['categorical'],chain_z[0])

        chain_z[0] = xh  # Overwrite last frame with the resulting x and h.
        chain_flat = chain_z.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat