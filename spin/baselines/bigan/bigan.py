import torch
from einops import rearrange
from torch import nn
from torch.nn import BCELoss
from .layers import Generator, Discriminator


class BiGAN(nn.Module):

    def __init__(self, input_size: int, n_nodes: int, hidden_size: int = 64):
        super().__init__()

        # networks
        self.bce = BCELoss()
        self.generator = Generator(input_size, n_nodes, hidden_size)
        self.discriminator = Discriminator(n_nodes, hidden_size)

    def forward(self, x, mask=None):
        return self.generator(x, mask)

    def adversarial_loss(self, prob_mat, mask):
        mask = rearrange(mask, 'b s n c -> b s (n c)')
        loss = torch.mul((1 - mask), torch.log(1 - prob_mat)).mean()
        return loss

    def bce_loss(self, inputs, mask):
        mask = rearrange(mask, 'b s n c -> b s (n c)')
        loss = self.bce(inputs, mask.float())
        return loss

    def consistency_loss(self, imp_fwd, imp_bwd):
        loss = 0.1 * torch.abs(imp_fwd - imp_bwd).mean()
        return loss

    @staticmethod
    def add_model_specific_args(parser):
        parser.opt_list('--hidden-size', type=int, tunable=True, default=32,
                        options=[32, 64, 128, 256])
        parser.add_argument('--u-size', type=int, default=None)
        return parser
