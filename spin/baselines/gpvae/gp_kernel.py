import torch

''' 
GP kernel functions 
'''

def rbf_kernel(T, length_scale):
    xs = torch.arange(0, T, dtype=torch.float32)
    xs_repeat = torch.repeat_interleave(xs.unsqueeze(1), T, dim=1)
    distance_matrix = (xs - xs_repeat)**2
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


def diffusion_kernel(T, length_scale):
    assert length_scale < 0.5, "length_scale has to be smaller than 0.5 for the "\
                               "kernel matrix to be diagonally dominant"
    sigmas = torch.ones([T, T]) * length_scale
    sigmas_tridiag = sigmas - (sigmas.triu(2) + sigmas.tril(-2))
    kernel_matrix = sigmas_tridiag + torch.eye(T)*(1. - length_scale)
    return kernel_matrix


def matern_kernel(T, length_scale):
    xs = torch.arange(1, T, dtype=torch.float32)
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = torch.abs(xs_in - xs_out)
    distance_matrix_scaled = distance_matrix / torch.sqrt(length_scale)
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


def cauchy_kernel(T, sigma, length_scale):
    xs = torch.arange(0, T, dtype=torch.float32)
    xs_repeat = torch.repeat_interleave(xs.unsqueeze(1), T, dim=1)
    distance_matrix = (xs - xs_repeat)**2
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = torch.divide(sigma, (distance_matrix_scaled + 1.))
    alpha = 0.001
    eye = torch.eye(n=kernel_matrix.shape[-1])
    return kernel_matrix + alpha * eye
