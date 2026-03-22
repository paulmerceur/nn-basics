import torch
import torch.nn as nn

from model import MLP
from MNIST import get_loaders


EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3


def train(epoch):
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        batch_loss = loss_fn(model(X), y)
        batch_loss.backward()
        optimizer.step()


def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(1) == y).sum().item()
    return correct / len(test_loader.dataset)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)

    model = MLP(hidden_sizes=[128]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        acc = evaluate()
        print(f"epoch {epoch:>2} | test acc {acc:.4f}")
