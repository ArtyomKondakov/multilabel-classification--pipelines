"""Lighting module."""
import pytorch_lightning as pl
import torch
from timm import create_model

from src.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.utils import load_object


class PosterModule(pl.LightningModule):  # noqa: WPS214
    """Class module.

    Args:
        pl (_type_): LightningModule
    """
    def __init__(self, config: Config):
        """Initialize an instance of the class..

        Args:
            config (Config): config
        """
        super().__init__()
        self._config = config

        self._model = create_model(
            num_classes=self._config.num_classes,
            **self._config.model_kwargs,
        )
        self._losses = get_losses(self._config.losses)
        metrics = get_metrics(
            num_classes=self._config.num_classes,
            num_labels=self._config.num_classes,
            task='multilabel',
            average='macro',
            threshold=0.5,
        )
        self._valid_metrics = metrics.clone(prefix='val_')
        self._test_metrics = metrics.clone(prefix='test_')

        self.save_hyperparameters()

    def forward(self, x_im: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x_im (Tensor): im

        Returns:
            torch.Tensor: pred
        """
        return self._model(x_im)

    def configure_optimizers(self):
        """Configure optimizers.

        Returns:
            _type_: _description_
        """
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(
            self._config.scheduler,
        )(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):
        """We count the loss for train.

        Args:
            batch (_type_): size batch
            batch_idx (_type_): idx batch

        Returns:
            _type_:  train loss
        """
        images, gt_labels = batch
        pr_logits = self(images)
        return self._calculate_loss(pr_logits, gt_labels, 'train_')

    def validation_step(self, batch, batch_idx):
        """We count the loss and metrics for validation.

        Args:
            batch (_type_): size batch
            batch_idx (_type_): idx batch
        """
        images, gt_labels = batch
        pr_logits = self(images)
        self._calculate_loss(pr_logits, gt_labels, 'val_')
        pr_labels = torch.sigmoid(pr_logits)
        self._valid_metrics(pr_labels, gt_labels)

    def test_step(self, batch, batch_idx):
        """We count the loss and metrics for test.

        Args:
            batch (_type_): size batch
            batch_idx (_type_): idx batch
        """
        images, gt_labels = batch
        pr_logits = self(images)
        pr_labels = torch.sigmoid(pr_logits)
        self._test_metrics(pr_labels, gt_labels)

    def on_validation_epoch_start(self) -> None:
        """On validation epoch start."""
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """On validation epoch end."""
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """On test epoch end."""
        self.log_dict(self._test_metrics.compute(), on_epoch=True)

    def _calculate_loss(
        self,
        pr_logits: torch.Tensor,
        gt_labels: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        total_loss = 0
        for cur_loss in self._losses:
            loss = cur_loss.loss(pr_logits, gt_labels)
            total_loss += cur_loss.weight * loss
            pref_cur_loss = '{prefix}{cur_loss}_loss'.format(
                prefix=prefix,
                cur_loss=cur_loss.name,
            )
            self.log(pref_cur_loss, loss.item())
        self.log('{prefix}total_loss'.format(prefix=prefix), total_loss.item())
        return total_loss
