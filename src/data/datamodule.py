from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from src.data.dataset import MyDataset
import pandas as pd


class MyDatamodule(LightningDataModule):
    def __init__(self, images_path=None, labels_path=None, fold=0, size=None, augmentation_p=False, train_batch_size=32,
                 val_batch_size=8, num_workers=8, **kwargs):
        super().__init__()
        self.images_path = images_path
        self.labels_path = labels_path
        self.labels = pd.read_csv(labels_path)
        self.fold = fold
        self.size = size
        self.augmentation_p = augmentation_p
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        labels_train = self.labels[self.labels["fold"] != self.fold]
        labels_val = self.labels[self.labels["fold"] == self.fold]
        train_dataset = MyDataset(
            images_path=self.images_path,
            labels=labels_train,
            size=self.size,
            augmentation_p=self.augmentation_p,
        )
        val_dataset = MyDataset(
            images_path=self.images_path,
            labels=labels_val,
            size=self.size,
            augmentation_p=False,
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )