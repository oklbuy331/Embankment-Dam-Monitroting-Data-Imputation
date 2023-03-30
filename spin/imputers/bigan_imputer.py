from tsl.imputers import Imputer


class BiGANImputer(Imputer):

    def shared_step(self, batch, mask):
        y = y_loss = batch.y
        y_hat = y_hat_loss = self.predict_batch(batch, postprocess=not self.scale_target)

        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        y_hat_loss, y_loss, mask = self.trim_warm_up(y_hat_loss, y_loss, mask)

        imputation, predictions = y_hat_loss
        imp_fwd, imp_bwd = predictions[:2]
        y_hat = y_hat[0]

        loss = sum([self.loss_fn(pred, y_loss, mask) for pred in predictions])
        loss += self.model.consistency_loss(imp_fwd, imp_bwd)

        return y_hat.detach(), y, loss

    def training_step(self, batch, batch_idx, optimizer_idx):

        # calculate reconstruction loss and discriminator probabilistic matrix
        y_hat, y, g_loss = self.shared_step(batch, batch.original_mask)
        d_prob = self.model.discriminator(y_hat, batch.mask)

        # train generator
        if optimizer_idx == 0:
            # generator loss is reconstruction loss + consistency loss + adversarial loss
            g_loss += self.model.adversarial_loss(d_prob, batch.mask)

            # Logging
            self.train_metrics.update(y_hat, y, batch.eval_mask)
            self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
            self.log_loss('train_g', g_loss, batch_size=batch.batch_size)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # discriminator loss is binary cross-entropy
            d_loss = self.model.bce_loss(d_prob, batch.mask)

            # Logging
            self.train_metrics.update(y_hat, y, batch.eval_mask)
            self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
            self.log_loss('train_d', d_loss, batch_size=batch.batch_size)
            return d_loss

    def configure_optimizers(self):
        opt_g = self.optim_class(self.model.generator.parameters(), **self.optim_kwargs)
        opt_d = self.optim_class(self.model.discriminator.parameters(), **self.optim_kwargs)
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop('monitor', None)
            scheduler_g = self.scheduler_class(opt_g, **self.scheduler_kwargs)
            scheduler_d = self.scheduler_class(opt_d, **self.scheduler_kwargs)
        return (
            {"optimizer": opt_g, "lr_scheduler": scheduler_g},
            {"optimizer": opt_d, "lr_scheduler": scheduler_d}
        )
