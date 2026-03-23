from typing import Literal

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081


def get_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    augment: Literal["default", "affine", "elastic"] = "default",
) -> tuple[DataLoader, DataLoader]:
    train_transforms = []
    if augment == "affine":
        train_transforms.append(
            transforms.RandomAffine(
                degrees=20,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10,
            )
        )
    elif augment == "elastic":
        train_transforms.append(
            transforms.ElasticTransform(alpha=34.0, sigma=4.0)
        )
    elif augment != "default":
        raise ValueError(f"Unknown augment mode: {augment}")

    train_transform = transforms.Compose(train_transforms + [
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])

    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader