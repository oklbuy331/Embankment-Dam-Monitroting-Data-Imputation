import math
import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions import Independent, Normal
from torch.nn import Conv1d, Linear
import torch.nn.functional as F


# Encoder

class JointEncoder(nn.Module):
    def __init__(self, z_size, n_features=20, hidden_sizes=(64, 64),
                 window_size=24, transpose=False, **kwargs):
        """ Encoder with 1d-convolutional network and factorized Normal posterior
            Used by joint VAE and HI-VAE with Standard Normal prior or GP-VAE with factorized Normal posterior
            :param z_size: latent space dimensionality
            :param n_features: number of features in each batch
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
            :param window_size: kernel size for Conv1D layer
            :param transpose: True for GP prior | False for Standard Normal prior
        """
        super(JointEncoder, self).__init__()
        self.z_size = int(z_size)
        self.conv1d = Conv1d(n_features, hidden_sizes[0],               # input tensor shape [batchs, n_features, steps]
                        kernel_size=window_size, padding="same", dtype=torch.float32)        # output tensor shape [batchs, n_filters, steps]
        self.dense1 = Linear(hidden_sizes[0], hidden_sizes[1], dtype=torch.float32)  # input tensor shape [batchs, steps, 128]
        self.dense2 = Linear(hidden_sizes[-1], 2*z_size, dtype=torch.float32)    # output tensor shape [batchs, steps, 70]
        self.transpose = transpose
        # return nn.Sequential(
        #             conv1d,
        #             dense1,
        #             nn.ReLU(),
        #             dense2,
        #             nn.ReLU()
        # )

    def forward(self, x):
        x = F.relu(self.dense1(self.conv1d(x.transpose(1, 2)).transpose(1, 2)))
        x = F.relu(self.dense2(x)).transpose(1, 2)

        if self.transpose:
            num_dim = len(x.shape.as_list())
            perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
            mapped = torch.transpose(x, perm=perm)
            return Independent(Normal(loc=mapped[..., :self.z_size, :],
                                      scale=F.softplus(mapped[..., self.z_size:, :])), 1)
        return Independent(Normal(loc=x[..., :self.z_size, :], scale=F.softplus(x[..., self.z_size:, :])), 1)


# Decoders

class GaussianDecoder(nn.Module):
    def __init__(self, output_size, hidden_sizes, z_size):
        """ Decoder parent class with no specified output distribution
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super().__init__()
        self.dense1 = Linear(z_size, hidden_sizes[0], dtype=torch.float32)
        self.dense2 = Linear(hidden_sizes[0], hidden_sizes[1], dtype=torch.float32)
        self.output_layer = Linear(hidden_sizes[1], output_size, dtype=torch.float32)

    def forward(self, z):
        z = F.relu(self.dense1(z))
        z = F.relu(self.dense2(z))
        mean = self.output_layer(z)
        var = torch.ones(mean.shape, dtype=torch.float32, device=mean.device)
        return Normal(loc=mean, scale=var)
