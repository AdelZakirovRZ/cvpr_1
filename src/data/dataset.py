from torch.utils.data import Dataset
import cv2
import os
import pandas as pd
from src.data.transforms import augment, preprocess
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, images_path, names, labels_path, size=(224, 224), augmentation_p=0.5):
        self.images_path = images_path
        self.names = names
        self.labels_path = labels_path
        self.labels = pd.read_csv(labels_path)
        self.size = size
        self.augmentation_p = augmentation_p

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.images_path, self.names[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augmentation_p > 0:
            img = augment(img, self.augmentation_p)
        img = preprocess(img, self.size)
        label = self.labels["label"].iloc[idx]
        return img, label
    
    def show_example(self, idx):
        img, label = self.__getitem__(idx)
        print(f"Label: {label}")
        img = img.numpy().transpose(1, 2, 0)
        img = img*[0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = img*255
        plt.imshow(img.astype("uint8"))
        plt.show()
