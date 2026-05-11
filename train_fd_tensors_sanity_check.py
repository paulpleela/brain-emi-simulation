"""
Train classification on FD tensor data for sanity check.
FD tensors have shape (512, 90) - spatial-frequency matrix representation.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

class FDTensorClassifier(nn.Module):
    def __init__(self):
        super(FDTensorClassifier, self).__init__()
        # Input shape: (Batch, 1, 512, 90)
        # Treat as a 2D image with height=512, width=90
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_and_evaluate(dataset_path, epochs=30, batch_size=16):
    print(f"\n{'='*50}\nEvaluating FD Tensor Dataset\n{'='*50}")
    
    # Load dataset
    data = np.load(dataset_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"Train positive ratio: {y_train.mean():.4f}")
    print(f"Val positive ratio: {y_val.mean():.4f}")
    
    # Convert to PyTorch tensors
    # Shape is (N, 512, 90) → add channel dimension: (N, 1, 512, 90)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    # ── NORMALISATION ──
    # Log scale and standardise
    X_train_t = torch.log1p(torch.abs(X_train_t))
    X_val_t = torch.log1p(torch.abs(X_val_t))
    
    mean = X_train_t.mean()
    std = X_train_t.std()
    if std == 0: std = 1e-8
    X_train_t = (X_train_t - mean) / std
    X_val_t = (X_val_t - mean) / std
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = FDTensorClassifier()
    
    # ── CLASS BALANCE WEIGHTING ──
    pos_cases = (y_train_t == 1).sum().float()
    neg_cases = (y_train_t == 0).sum().float()
    pos_weight = neg_cases / pos_cases if pos_cases > 0 else torch.tensor(1.0)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training Loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss = {train_loss/len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_outputs = torch.sigmoid(val_logits)
        val_preds = (val_outputs >= 0.5).float()
        
        acc = accuracy_score(y_val_t.numpy(), val_preds.numpy())
        auc = roc_auc_score(y_val_t.numpy(), val_outputs.numpy())
        
        print(f"\nFinal Validation Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    
    return acc, auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    dataset_path = "deep_learning/datasets/dataset_fd_tensors.npz"
    if os.path.exists(dataset_path):
        acc, auc = train_and_evaluate(dataset_path, epochs=args.epochs)
    else:
        print(f"Dataset not found: {dataset_path}")
