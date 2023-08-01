from torch.utils.data import Dataset
import cv2
import os
import pandas as pd
from src.data.transforms import augment, preprocess
import matplotlib.pyplot as plt

CLASS2LABEL = {
    0: "NPK_",
    1: "N_KCa",
    2: "NP_Ca",
    3: "NPKCa",
    4: "unfertilized",
    5: "_PKCa",
    6: "NPKCa+m+s",
}

LABEL2CLASS = {
    "NPK_": 0,
    "N_KCa": 1,
    "NP_Ca": 2,
    "NPKCa": 3,
    "unfertilized": 4,
    "_PKCa": 5,
    "NPKCa+m+s": 6,
}


class MyDataset(Dataset):
    """
    Custom dataset class.
    images_path: str, path to images directory
    names: list of str, image names
    labels_path: str, path to labels csv file
    size: tuple (height, width), size of image after preprocessing
    augmentation_p: float, probability of applying augmentation
    """

    def __init__(
        self, images_path, labels, size=(224, 224), augmentation_p=0.5
    ):
        self.images_path = images_path
        self.labels = labels
        self.size = size
        self.augmentation_p = augmentation_p

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns image and label at index idx.
        idx: int
        return: torch tensor, int
        """
        name, class_ = self.labels.iloc[idx][["names", "classes"]]
        img = cv2.imread(os.path.join(self.images_path, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augmentation_p > 0:
            img = augment(img, self.size, self.augmentation_p)
        img = preprocess(img, self.size)
        return img, class_

    def show_example(self, idx):
        """
        Shows example image and label at index idx.
        idx: int
        """
        img, class_ = self.__getitem__(idx)
        label = CLASS2LABEL[class_]
        print(f"Label: {label}")
        img = img.numpy().transpose(1, 2, 0)
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = img * 255
        plt.imshow(img.astype("uint8"))
        plt.show()
