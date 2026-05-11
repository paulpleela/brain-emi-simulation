#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-6


class PortMatrixCNN2DClassifier(nn.Module):
    def __init__(self, in_channels: int, dropout_rate: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def compute_train_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, EPSILON)
    return mean, std


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mean) / std).astype(np.float32)


def to_port_cnn_input(batch_x: torch.Tensor) -> torch.Tensor:
    batch_x = batch_x.permute(0, 1, 4, 2, 3).contiguous()
    batch_size, num_frequencies, num_channels, height, width = batch_x.shape
    return batch_x.reshape(batch_size, num_frequencies * num_channels, height, width)


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    predictions = (probabilities >= 0.5).astype(np.float32)
    accuracy = accuracy_score(labels, predictions)
    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = float("nan")
    return float(accuracy), float(auc)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for batch_x, batch_y in loader:
        batch_x = to_port_cnn_input(batch_x.to(DEVICE))
        batch_y = batch_y.to(DEVICE).float().view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel())
        all_labels.extend(batch_y.detach().cpu().numpy().ravel())

    all_labels_np = np.asarray(all_labels)
    all_probs_np = np.asarray(all_probs)
    accuracy, auc = compute_metrics(all_labels_np, all_probs_np)
    return total_loss / len(all_labels_np), accuracy, auc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for batch_x, batch_y in loader:
        batch_x = to_port_cnn_input(batch_x.to(DEVICE))
        batch_y = batch_y.to(DEVICE).float().view(-1, 1)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += loss.item() * len(batch_y)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
        all_labels.extend(batch_y.cpu().numpy().ravel())

    all_labels_np = np.asarray(all_labels)
    all_probs_np = np.asarray(all_probs)
    accuracy, auc = compute_metrics(all_labels_np, all_probs_np)
    return total_loss / len(all_labels_np), accuracy, auc


def train_and_evaluate(dataset_path, epochs=20, batch_size=16):
    print(f"\n{'='*50}\nEvaluating Dataset: {os.path.basename(dataset_path)}\n{'='*50}")

    data = np.load(dataset_path)
    x_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    x_val = data["X_val"].astype(np.float32) if "X_val" in data else x_train
    y_val = data["y_val"].astype(np.float32) if "y_val" in data else y_train
    x_test = data["X_test"].astype(np.float32) if "X_test" in data else None
    y_test = data["y_test"].astype(np.float32) if "y_test" in data else None

    mean, std = compute_train_stats(x_train)
    x_train = normalize(x_train, mean, std)
    x_val = normalize(x_val, mean, std)
    if x_test is not None:
        x_test = normalize(x_test, mean, std)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train).view(-1, 1)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val).view(-1, 1)
    x_test_t = torch.from_numpy(x_test) if x_test is not None else None
    y_test_t = torch.from_numpy(y_test).view(-1, 1) if y_test is not None else None

    print(f"  Raw tensor shape: {tuple(data['X_train'].shape)}")
    print(f"  Normalization mean/std: {mean.shape} / {std.shape}")
    print(f"  CNN input shape: {(x_train_t.shape[0], x_train_t.shape[1] * x_train_t.shape[-1], x_train_t.shape[2], x_train_t.shape[3])}")

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    model = PortMatrixCNN2DClassifier(in_channels=x_train_t.shape[1] * x_train_t.shape[-1]).to(DEVICE)

    pos_cases = (y_train_t == 1).sum().float()
    neg_cases = (y_train_t == 0).sum().float()
    pos_weight = neg_cases / pos_cases if pos_cases > 0 else torch.tensor(1.0)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_val_auc = float("-inf")
    best_epoch = 0
    best_state = None
    best_train_auc = float("nan")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_auc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_train_auc = train_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  Epoch {epoch:2d}/{epochs}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"train_auc={train_auc:.4f} val_auc={val_auc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion)
    test_acc = None
    test_auc = None
    if x_test_t is not None and y_test_t is not None:
        test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=batch_size, shuffle=False)
        _, test_acc, test_auc = evaluate(model, test_loader, criterion)

    print(f"  Best epoch: {best_epoch}")
    print(f"  Best train AUC: {best_train_auc:.4f}")
    print(f"  Best val AUC:   {best_val_auc:.4f}")
    if test_auc is not None:
        print(f"  Test AUC:       {test_auc:.4f}")

    return val_acc, val_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dir", type=str, default="datasets")
    args = parser.parse_args()

    formats = ["real_imag", "mag_phase", "mag_only"]
    results = {}

    for fmt in formats:
        path = os.path.join(args.dir, f"dataset_{fmt}.npz")
        if os.path.exists(path):
            acc, auc = train_and_evaluate(path, epochs=args.epochs)
            results[fmt] = {"acc": acc, "auc": auc}
        else:
            print(f"Dataset not found: {path}")

    print("\n" + "=" * 40)
    print("FINAL SUMMARY")
    print("=" * 40)
    for fmt, mets in results.items():
        print(f"{fmt.ljust(15)}: ACC = {mets['acc']:.4f}, AUC = {mets['auc']:.4f}")