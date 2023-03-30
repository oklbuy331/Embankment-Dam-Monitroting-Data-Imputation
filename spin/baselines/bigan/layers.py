import numpy as np
import torch
from einops import rearrange
from torch import nn
from tsl.nn.functional import reverse_tensor

from ..brits.layers import RITS, TemporalDecay


class Generator(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, hidden_size: int = 64):
        super().__init__()
        self.n_nodes = n_nodes
        self.rits_fwd = RITS(input_size * n_nodes, hidden_size)
        self.rits_bwd = RITS(input_size * n_nodes, hidden_size)

    def forward(self, x, mask=None):
        # tsl shape to original shape: [b s n c] -> [b s c]
        x = rearrange(x, 'b s n c -> b s (n c)')
        mask = rearrange(mask, 'b s n c -> b s (n c)')
        # forward
        imp_fwd, pred_fwd = self.rits_fwd(x, mask)
        # backward
        x_bwd = reverse_tensor(x, dim=1)
        mask_bwd = reverse_tensor(mask, dim=1) if mask is not None else None
        imp_bwd, pred_bwd = self.rits_bwd(x_bwd, mask_bwd)
        imp_bwd, pred_bwd = reverse_tensor(imp_bwd, dim=1), \
                            [reverse_tensor(pb, dim=1) for pb in pred_bwd]
        # stack into shape = [batch, directions, steps, features]
        imputation = (imp_fwd + imp_bwd) / 2
        predictions = [imp_fwd, imp_bwd] + pred_fwd + pred_bwd

        imputation = rearrange(imputation, 'b s (n c) -> b s n c',
                               n=self.n_nodes)
        predictions = [rearrange(pred, 'b s (n c) -> b s n c', n=self.n_nodes)
                       for pred in predictions]

        return imputation, predictions


class DiscriminativeLayer(nn.Module):

    def __init__(self, n_nodes: int, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.LSTMCell(2 * n_nodes, hidden_size)
        self.disc_prob_model = nn.Sequential(
            nn.Linear(2 * n_nodes, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, n_nodes),
            nn.Sigmoid()
        )

    # return discriminative probabilistic matrix calculated from generator output and mask
    def forward(self, x, mask=None, p_hint=0.8):
        # x : [batch, steps, features]
        batch_size = x.shape[0]
        steps = x.shape[1]
        n_nodes = x.shape[2]
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)
        hint = self.sample_hint(p_hint, batch_size, steps, n_nodes).to(mask.device)
        hint_matrix = mask * hint
        # initialize LSTM states
        h = self.init_hidden_states(x)
        c = self.init_hidden_states(x)
        # concatenate hint and data
        inputs = torch.cat((x, hint_matrix), dim=2)

        d_prob_matrix = []
        for step in range(steps):
            xh = inputs[:, step, :]
            d_prob = self.disc_prob_model(xh)
            h, c = self.rnn_cell(xh, (h, c))
            d_prob_matrix.append(d_prob)

        # list -> [batch, steps, features]
        d_prob_matrix = torch.stack(d_prob_matrix, dim=-2)
        return d_prob_matrix

    def init_hidden_states(self, x):
        return torch.zeros((x.shape[0], self.hidden_size)).to(x.device)

    # sample hint matrix every batch
    @staticmethod
    def sample_hint(p, batch_size=32, rows=24, cols=20):
        # mini-batch of mask matrix of shape: [batch, steps, n_nodes]
        unif_random_matrix = torch.rand(size=[batch_size, rows, cols])
        binary_random_matrix = 1. * (unif_random_matrix < p)
        return binary_random_matrix


class Discriminator(nn.Module):
    def __init__(self, n_nodes: int, hidden_size: int = 64):
        super().__init__()
        self.disc_fwd = DiscriminativeLayer(n_nodes, hidden_size)
        self.disc_bwd = DiscriminativeLayer(n_nodes, hidden_size)

    def forward(self, x, mask):
        # tsl shape to original shape: [b s n c] -> [b s c]
        x = rearrange(x, 'b s n c -> b s (n c)')
        mask = rearrange(mask, 'b s n c -> b s (n c)')

        # forward
        d_prob_fwd = self.disc_fwd(x, mask)

        # backward
        x_bwd = reverse_tensor(x, dim=1)
        mask_bwd = reverse_tensor(mask, dim=1) if mask is not None else None
        d_prob_bwd = self.disc_bwd(x_bwd, mask_bwd)
        d_prob_bwd = reverse_tensor(d_prob_bwd, dim=1)

        # combine forward and backward discriminative probability
        d_prob = (d_prob_fwd + d_prob_bwd) / 2
        return d_prob

    @staticmethod
    def adversarial_loss(imp_fwd, imp_bwd):
        loss = 0.1 * torch.abs(imp_fwd - imp_bwd).mean()
        return loss

