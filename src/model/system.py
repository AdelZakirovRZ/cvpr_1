import pytorch_lightning as pl
import timm
import torch
from src.data.mix_up import mixup_data, mixup_criterion
from torchmetrics.classification import MulticlassAccuracy
from src.data.dataset import CLASS2LABEL, LABEL2CLASS
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model.utils import top_k_accuracy
import torchvision


class System(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = timm.create_model(**self.hparams.model)
        # if hparams.load_checkpoint_file:
        #     self.model = System.load_from_checkpoint(
        #         hparams.load_checkpoint_file
        #     ).model
        self.criterion = torch.nn.functional.cross_entropy
        self.acc_train = MulticlassAccuracy(num_classes=7, top_k=1, average='none')
        self.acc_val = MulticlassAccuracy(num_classes=7, top_k=1, average='none')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.hparams.mixup:
            x, y_a, y_b, lam = mixup_data(x, y, self.hparams['mixup_alpha'], True)
        preds = self.model(x)
        if self.hparams.mixup:
            loss = mixup_criterion(self.criterion, preds, y_a, y_b, lam)
        else:
            loss = self.criterion(preds, y)
        probs = torch.softmax(preds, dim=1)
        acc = top_k_accuracy(preds, y)[0]
        self.log("train_acc_topk", acc, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.acc_train.update(probs, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss

    def on_train_epoch_end(self):
        scores = self.acc_train.compute()
        self.acc_train.reset()
        logs = {f"train_acc_{CLASS2LABEL[i]}": scores[i] for i in range(len(scores))}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.acc_train.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.criterion(preds, y)
        probs = torch.softmax(preds, dim=1)
        acc = top_k_accuracy(preds, y)[0]
        self.log("val_acc_topk", acc, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.acc_val.update(probs, y)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss

    def on_validation_epoch_end(self):
        scores = self.acc_val.compute()
        logs = {f"val_acc_{CLASS2LABEL[i]}": scores[i] for i in range(len(scores))}
        logs.update({"val_acc": sum(scores) / 7})
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.acc_val.reset()

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        T = len(self.trainer.train_dataloader)
        optimizer = None
        if self.hparams.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == 'Sophia':
            optimizer = torch.optim.Sophia(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        cheduler_dict = None
        if self.hparams.scheduler == 'Cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=T, eta_min=self.hparams.lr_end)
            cheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [cheduler_dict]
