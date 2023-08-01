import os
from argparse import Namespace
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from src.data.datamodule import MyDatamodule
from src.model.system import System
from src.experiment_config import experiment_config_cmdline


def train(hparams: Namespace):
    dm = MyDatamodule(**vars(hparams))
    system = System(hparams)
    checkpoint_names = "cp_{epoch:02d}_{val_loss:.3f}_{val_acc:.3f}"
    checkpoints_path = os.path.join(
        hparams.checkpoints_path, hparams.experiment_name, hparams.run_name
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_path,
        filename=checkpoint_names,
        save_top_k=1,
        verbose=True,
        monitor="val_acc",
        mode="max",
        save_weights_only=False,
    )
    es_callback = EarlyStopping(
        monitor="val_loss",
        patience=hparams.es_patience,
        mode="min",
        verbose=True,
        strict=False,
    )
    mlf_logger = MLFlowLogger(
        experiment_name=hparams.experiment_name, tracking_uri=hparams.tracking_uri
    )
    mlf_logger.tags = {
        "mlflow.runName": hparams.run_name,
        "mlflow.note.content": hparams.note,
    }
    trainer = Trainer(
        max_epochs=hparams.epochs,
        devices=hparams.gpus,
        callbacks=[es_callback, checkpoint_callback],
        num_nodes=1,
        precision=32,
        accumulate_grad_batches=hparams.acc_grad,
        num_sanity_val_steps=0,
        logger=mlf_logger,
        strategy=hparams.strategy,
    )
    trainer.fit(model=system, datamodule=dm)


if __name__ == "__main__":
    train(experiment_config_cmdline())
