from tqdm import tqdm


__all__ = ['_stage0_epoch', 'run_stage0_epoch']


def _stage0_epoch(self, epoch: int):
    self.model.train()
    total_loss = 0.0
    total_rec_v1 = 0.0
    total_rec_v2 = 0.0

    progress = tqdm(
        self.train_loader,
        desc=f"Stage0 {epoch}",
        leave=False,
        disable=not self._show_batch_progress(),
    )
    for batch in progress:
        x = self._prepare_batch(batch)

        outputs = self.model(x, stage="stage0")
        rec_v1, rec_v2 = self._dual_reconstruction_losses_from_outputs(outputs, x)
        loss_rec = 0.5 * (rec_v1 + rec_v2) if self._is_dual_view_model() else rec_v1
        loss = loss_rec

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()
        total_rec_v1 += rec_v1.item()
        total_rec_v2 += rec_v2.item()
        progress.set_postfix(
            {
                "r1": f"{rec_v1.item():.4f}",
                "r2": f"{rec_v2.item():.4f}",
            }
        )

    avg_loss = total_loss / max(1, len(self.train_loader))
    logs = {"loss": avg_loss}
    if self._is_dual_view_model():
        logs["rec_v1"] = total_rec_v1 / max(1, len(self.train_loader))
        logs["rec_v2"] = total_rec_v2 / max(1, len(self.train_loader))
    self._log_epoch("Stage0", epoch, self.config.epoch_stage0, logs)


def run_stage0_epoch(solver, loader, epoch):
    if loader is not None and loader is not solver.train_loader:
        original_loader = solver.train_loader
        solver.train_loader = loader
        try:
            return solver._stage0_epoch(epoch)
        finally:
            solver.train_loader = original_loader
    return solver._stage0_epoch(epoch)
