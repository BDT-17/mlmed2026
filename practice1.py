import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

device = "cpu"

batch_size = 64
epochs = 30
lr = 1e-4


class Model(nn.Module):
    def __init__(self, d_model, labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, labels),
        )

    def forward(self, x):
        return self.net(x)


class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # Load data
    train = np.loadtxt(r"D:\DS\ML in medicine\practical_1\mitbih_train.csv", delimiter=",")
    test = np.loadtxt(r"D:\DS\ML in medicine\practical_1\mitbih_test.csv", delimiter=",")

    X_train = torch.tensor(train[:, :-1], dtype=torch.float32)
    y_train = torch.tensor(train[:, -1], dtype=torch.long)

    X_test = torch.tensor(test[:, :-1], dtype=torch.float32)
    y_test = torch.tensor(test[:, -1], dtype=torch.long)

    train_loader = DataLoader(
        ECGDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  
    )

    val_loader = DataLoader(
        ECGDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = Model(187, 5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {acc:.4f} | "
            f"Val F1: {f1:.4f}"
        )


