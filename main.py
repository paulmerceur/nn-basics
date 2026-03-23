import torch
import torch.nn as nn
from itertools import product

from model import MLP
from MNIST import get_loaders


BATCH_SIZE = 64
LR = 1e-3
TARGET_ACC = 0.99
PATIENCE = 5
SEED = 0
HIDDEN_SIZE_CANDIDATES = [1024, 2048, 4096]
TRICK_OPTIONS = {
    "augment": ["default", "affine", "elastic"],
    "batch_norm": [True],
}


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        batch_loss = loss_fn(model(X), y)
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item() * X.size(0)
        n_samples += X.size(0)

    return total_loss / n_samples


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(1) == y).sum().item()

    return correct / len(test_loader.dataset)


def build_trick_configs():
    option_names = list(TRICK_OPTIONS)
    option_values = [TRICK_OPTIONS[name] for name in option_names]
    return [
        dict(zip(option_names, values))
        for values in product(*option_values)
    ]


def format_trick_config(config):
    return " | ".join(
        f"{name}={'on' if value else 'off'}" if isinstance(value, bool) else f"{name}={value}"
        for name, value in config.items()
    )


def run_trial(hidden_size, trick_config, device):
    torch.manual_seed(SEED)
    train_loader, test_loader = get_loaders(
        batch_size=BATCH_SIZE,
        augment=trick_config["augment"],
    )

    model = MLP(
        hidden_sizes=[hidden_size],
        activation=nn.ReLU,
        batch_norm=trick_config["batch_norm"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_epoch = 0
    epoch = 0
    stale_epochs = 0

    print(f"\nHidden layer: {hidden_size} neurons  |  {format_trick_config(trick_config)}")
    print(f"{'epoch':>5}  {'train loss':>10}  {'test acc':>8}  {'best':>8}  {'stale':>5}")
    print("-" * 47)

    while stale_epochs < PATIENCE:
        epoch += 1
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        acc = evaluate(model, test_loader, device)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            stale_epochs = 0
            marker = "  *"
        else:
            stale_epochs += 1
            marker = ""

        print(
            f"{epoch:5d}  {train_loss:10.4f}  {acc * 100:7.2f}%  "
            f"{best_acc * 100:7.2f}%  {stale_epochs:5d}{marker}"
        )

    return {
        "hidden_size": hidden_size,
        "tricks": trick_config,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trick_configs = build_trick_configs()

    print(
        f"MNIST sweep  |  target {TARGET_ACC * 100:.2f}%  |  patience {PATIENCE}  "
        f"|  batch {BATCH_SIZE}  |  lr {LR:g}  |  device {device}"
    )
    print(f"Candidates: {HIDDEN_SIZE_CANDIDATES}")
    print(f"Tricks: {[format_trick_config(config) for config in trick_configs]}")

    results = []

    for hidden_size in HIDDEN_SIZE_CANDIDATES:
        width_results = []

        for trick_config in trick_configs:
            result = run_trial(hidden_size, trick_config, device)
            results.append(result)
            width_results.append(result)

        successful_results = [
            result for result in width_results if result["best_acc"] >= TARGET_ACC
        ]
        if successful_results:
            best_result = max(successful_results, key=lambda result: result["best_acc"])
            print(
                f"\nReached target with {hidden_size} neurons using "
                f"{format_trick_config(best_result['tricks'])} "
                f"(best {best_result['best_acc'] * 100:.2f}% at epoch {best_result['best_epoch']})."
            )
            break
    else:
        best_result = max(results, key=lambda result: result["best_acc"])
        print(
            f"\nNo candidate reached {TARGET_ACC * 100:.2f}%. "
            f"Best was {best_result['hidden_size']} neurons with "
            f"{format_trick_config(best_result['tricks'])} at "
            f"{best_result['best_acc'] * 100:.2f}%."
        )
