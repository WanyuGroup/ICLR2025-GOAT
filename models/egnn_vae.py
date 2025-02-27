import torch
from torch.nn import functional as F
from utils import utilis_func as uf
from models.egnn_blocks import EGNN_encoder, EGNN_decoder
from models.equivariant_rectified_flow import sum_except_batch, gaussian_KL, gaussian_KL_for_dimension


class EnHierarchicalVAE(torch.nn.Module):
    """
    The E(n) Hierarchical VAE Module.
    """

    def __init__(
            self,
            encoder: EGNN_encoder,
            decoder: EGNN_decoder,
            in_node_nf: int, n_dims: int, latent_node_nf: int,
            kl_weight: float,
            norm_values=(1., 1., 1.), norm_biases=(None, 0., 0.),
            include_charges=True):
        super().__init__()

        self.include_charges = include_charges

        self.encoder = encoder
        self.decoder = decoder

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.latent_node_nf = latent_node_nf
        self.num_classes = self.in_node_nf - self.include_charges
        self.kl_weight = kl_weight

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def compute_reconstruction_error(self, xh_rec, xh):
        """Computes reconstruction error."""

        bs, n_nodes, dims = xh.shape

        # Error on positions.
        x_rec = xh_rec[:, :, :self.n_dims]
        x = xh[:, :, :self.n_dims]
        error_x = sum_except_batch((x_rec - x) ** 2)

        # Error on classes.
        h_cat_rec = xh_rec[:, :, self.n_dims:self.n_dims + self.num_classes]
        h_cat = xh[:, :, self.n_dims:self.n_dims + self.num_classes]
        h_cat_rec = h_cat_rec.reshape(bs * n_nodes, self.num_classes)
        h_cat = h_cat.reshape(bs * n_nodes, self.num_classes)
        error_h_cat = F.cross_entropy(h_cat_rec, h_cat.argmax(dim=1), reduction='none')
        error_h_cat = error_h_cat.reshape(bs, n_nodes, 1)
        error_h_cat = sum_except_batch(error_h_cat)
        # error_h_cat = sum_except_batch((h_cat_rec - h_cat) ** 2)

        # Error on charges.
        if self.include_charges:
            h_int_rec = xh_rec[:, :, -self.include_charges:]
            h_int = xh[:, :, -self.include_charges:]
            error_h_int = sum_except_batch((h_int_rec - h_int) ** 2)
        else:
            error_h_int = 0.

        error = error_x + error_h_cat + error_h_int

        if self.training:
            denom = (self.n_dims + self.in_node_nf) * xh.shape[1]
            error = error / denom

        return error

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def compute_loss(self, x, h, node_mask, edge_mask, context):
        """Computes an estimator for the variational lower bound."""

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)

        # Encoder output.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.encode(x, h, node_mask, edge_mask, context)

        # KL distance.
        # KL for invariant features.
        zeros, ones = torch.zeros_like(z_h_mu), torch.ones_like(z_h_sigma)
        loss_kl_h = gaussian_KL(z_h_mu, ones, zeros, ones, node_mask)
        # KL for equivariant features.
        assert z_x_sigma.mean(dim=(1, 2), keepdim=True).expand_as(z_x_sigma).allclose(z_x_sigma, atol=1e-7)
        zeros, ones = torch.zeros_like(z_x_mu), torch.ones_like(z_x_sigma.mean(dim=(1, 2)))
        subspace_d = self.subspace_dimensionality(node_mask)
        loss_kl_x = gaussian_KL_for_dimension(z_x_mu, ones, zeros, ones, subspace_d)
        loss_kl = loss_kl_h + loss_kl_x

        # Infer latent z.
        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        uf.assert_correctly_masked(z_xh_mean, node_mask)
        z_xh_sigma = torch.cat([z_x_sigma.expand(-1, -1, 3), z_h_sigma], dim=2)
        z_xh = self.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        # z_xh = z_xh_mean
        uf.assert_correctly_masked(z_xh, node_mask)
        uf.assert_mean_zero_with_mask(z_xh[:, :, :self.n_dims], node_mask)

        # Decoder output (reconstruction).
        x_recon, h_recon = self.decoder._forward(z_xh, node_mask, edge_mask, context)
        xh_rec = torch.cat([x_recon, h_recon], dim=2)
        loss_recon = self.compute_reconstruction_error(xh_rec, xh)

        # Combining the terms
        assert loss_recon.size() == loss_kl.size()
        loss = loss_recon + self.kl_weight * loss_kl

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {'loss_t': loss.squeeze(), 'rec_error': loss_recon.squeeze()}

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the ELBO if training. And if eval then always computes NLL.
        """

        loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context)

        neg_log_pxh = loss

        return neg_log_pxh

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = uf.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = uf.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.latent_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def encode(self, x, h, node_mask=None, edge_mask=None, context=None):
        """Computes q(z|x)."""

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)

        uf.assert_mean_zero_with_mask(xh[:, :, :self.n_dims], node_mask)

        # Encoder output.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.encoder._forward(xh, node_mask, edge_mask, context)

        bs, _, _ = z_x_mu.size()
        sigma_0_x = torch.ones(bs, 1, 1).to(z_x_mu) * 0.0032
        sigma_0_h = torch.ones(bs, 1, self.latent_node_nf).to(z_h_mu) * 0.0032

        return z_x_mu, sigma_0_x, z_h_mu, sigma_0_h

    def decode(self, z_xh, node_mask=None, edge_mask=None, context=None):
        """Computes p(x|z)."""

        # Decoder output (reconstruction).
        x_recon, h_recon = self.decoder._forward(z_xh, node_mask, edge_mask, context)
        uf.assert_mean_zero_with_mask(x_recon, node_mask)

        xh = torch.cat([x_recon, h_recon], dim=2)

        x = xh[:, :, :self.n_dims]
        uf.assert_correctly_masked(x, node_mask)

        h_int = xh[:, :, -1:] if self.include_charges else torch.zeros(0).to(xh)
        h_cat = xh[:, :, self.n_dims:-1]  # TODO: have issue when include_charges is False
        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}

        return x, h

    @torch.no_grad()
    def reconstruct(self, x, h, node_mask=None, edge_mask=None, context=None):
        pass

    def log_info(self):
        """
        Some info logging of the model.
        """
        info = None
        print(info)

        return info