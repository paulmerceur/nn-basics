from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081


def get_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])

    train_set = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader