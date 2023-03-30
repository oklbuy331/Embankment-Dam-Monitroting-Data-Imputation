import numpy as np
from einops import rearrange
import torch.nn as nn
from .layers import JointEncoder, GaussianDecoder
from .gp_kernel import *


class GPVAE(nn.Module):

    def __init__(self, latent_dims, n_nodes, window_size,
                 kernel="cauchy", sigma=1.005,
                 length_scale=1.0, kernel_scales=1,
                 encoder_sizes=(256, 256), encoder=JointEncoder,
                 decoder_sizes=(256, 256), decoder=GaussianDecoder,
                 image_preprocessor=None, beta=1.0, **kwargs):
        super().__init__()
        self.latent_dim = latent_dims
        self.time_length = window_size
        self.kernel = kernel
        self.sigma = sigma
        self.encoder = encoder(latent_dims, n_nodes, encoder_sizes, **kwargs)
        self.decoder = decoder(n_nodes, decoder_sizes, latent_dims)
        self.kl_loss = nn.KLDivLoss(reduce=False, log_target=True)
        self.preprocessor = image_preprocessor
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales
        self.beta = beta
        self.prior = None

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def compute_posterior(self, x):
        qz_x = self.encode(x)
        z = qz_x.rsample()
        z_mean = qz_x.mean
        px_z = self.decode(z.transpose(1, 2))
        postr_mean = self.decode(z_mean.transpose(1, 2)).mean
        return z, px_z, qz_x, postr_mean

    def forward(self, x, mask=None):
        # tsl shape to original shape: [b s n c] -> [b n s]
        x = rearrange(x, 'b s n c -> b n (s c)')
        if mask is not None:
            mask = rearrange(mask, 'b s n c -> b n (s c)')
        return self.decode(self.encode(x).sample()).sample()

    def get_prior(self):
        if self.prior is None:
            # Compute kernel matrices for each latent dimension, where every timestamp on time dimension is a rv.
            kernel_matrices = []
            for i in range(self.kernel_scales):
                if self.kernel == "rbf":
                    kernel_matrices.append(rbf_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "diffusion":
                    kernel_matrices.append(diffusion_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "matern":
                    kernel_matrices.append(matern_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "cauchy":
                    kernel_matrices.append(cauchy_kernel(self.time_length, self.sigma, self.length_scale / 2**i))

            # Combine kernel matrices for each latent dimension
            tiled_matrices = []
            total = 0
            for i in range(self.kernel_scales):
                if i == self.kernel_scales-1:
                    multiplier = self.latent_dim - total
                else:
                    multiplier = int(np.ceil(self.latent_dim / self.kernel_scales))
                    total += multiplier
                tiled_matrices.append(torch.tile(kernel_matrices[i].unsqueeze(0), [multiplier, 1, 1]))
            kernel_matrix_tiled = torch.cat(tiled_matrices).to('cuda')
            assert len(kernel_matrix_tiled) == self.latent_dim

            self.prior = torch.distributions.MultivariateNormal(
                loc=torch.zeros([self.latent_dim, self.time_length], dtype=torch.float32, device='cuda'),
                covariance_matrix=kernel_matrix_tiled)
        return self.prior

    def kl_div(self, input, target):
        div = self.kl_loss(input, target)
        return div

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--hidden-size', type=int, default=64)
        return parser
