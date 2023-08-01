import albumentations as A
from albumentations.pytorch import ToTensorV2


def augment(image, size, p=0.5):
    """Augment image with albumentations.
    image: numpy array
    p: probability of applying augmentation
    return: numpy array"""
    augmentation = A.Compose([
        A.Resize(size[0], size[1], always_apply=True),
        A.CoarseDropout(max_holes=16, max_height=64, min_height=4, max_width=64, min_width=4, p=p),
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomRotate90(p=p),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=p),
        A.GaussNoise(p=p),
        A.GaussianBlur(blur_limit=(3, 63), p=p),
        A.RandomBrightnessContrast(p=p),
        # A.RandomFog(p=p),
        # A.RandomRain(p=p),
        # A.RandomShadow(p=p),
        # A.RandomSunFlare(p=p),
        A.GridDistortion(p=p),
        A.Perspective(p=p),
        A.ISONoise(color_shift=(0.01, 0.09),
                   intensity=(0.1, 0.2), p=p),
    ], p=1)
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
