import albumentations as A
from albumentations.pytorch import ToTensorV2

def augment(image, p=0.5):
    """Augment image with albumentations.
    image: numpy array
    p: probability of applying augmentation
    return: numpy array"""
    augmentation = A.Compose([
        A.CoarseDropout(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.GaussNoise(p=0.5),
        A.GaussianBlur(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomFog(p=0.5),
        A.RandomRain(p=0.5),
        A.RandomShadow(p=0.5),
        A.RandomSunFlare(p=0.5),
        A.GridDistortion(p=0.5),
        A.Perspective(p=0.5),
    ], p=p)
    augmented = augmentation(image=image)
    return augmented['image']

def preprocess(image, size=(224, 224)):
    """
    Preprocess image for model input. Normalization is always using ImageNet stats.
    image: numpy array
    size: tuple (height, width)
    return: torch tensor
    """
    preprocessing = A.Compose([
        A.Resize(size[0], size[1]),
        A.Normalize(),
        ToTensorV2(),
    ])
    preprocessed = preprocessing(image=image)
    return preprocessed['image']
