import torch
from einops import rearrange
from tsl.imputers import Imputer
from ..baselines import GPVAE


class GPVAEImputer(Imputer):

    def shared_step(self, batch, mask, return_parts=False):

        y = batch.y
        x = rearrange(batch.x, 'b s n c -> b s (n c)')
        mask = rearrange(mask, 'b s n c -> b s (n c)')

        pz = self.model.get_prior()
        z, px_z, qz_x, postr_mean = self.model.compute_posterior(x)

        # compute negative log likelihood on observed point
        nll = -px_z.log_prob(x)
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        nll = torch.where(mask, nll, torch.zeros_like(nll))
        nll = torch.sum(nll, (1, 2))

        # compute KL divergence between Gaussian process prior and variational distribution
        kl_div = self.model.kl_div(qz_x.log_prob(z), pz.log_prob(z))
        kl_div = torch.sum(kl_div, dim=1)
        elbo = -nll - self.model.beta * kl_div
        loss = -torch.mean(elbo, dim=0)

        if return_parts:
            nll = torch.mean(nll)  # scalar
            kl_div = torch.mean(kl_div)  # scalar
            return nll, kl_div, postr_mean.detach().unsqueeze(-1), y, loss

        return postr_mean.detach().unsqueeze(-1), y, loss

    def training_step(self, batch, batch_idx):

        y_hat, y, loss = self.shared_step(batch, batch.original_mask)

        # Logging
        self.train_metrics.update(y_hat, y, batch.eval_mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):

        y_hat, y, val_loss = self.shared_step(batch, batch.mask)

        # Logging
        self.val_metrics.update(y_hat, y, batch.eval_mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)
        return val_loss
