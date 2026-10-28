"""Task 2: PyTorch Dataset/DataLoader, MLP, early stopping, model save/reload, sklearn baseline."""
import copy, os, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CsvDataset(Dataset):
    """Dataset backed by a CSV file containing numeric features and a 'target' column."""
    def __init__(self, csv_path):
        frame = pd.read_csv(csv_path)
        self.y = torch.tensor(frame["target"].to_numpy(), dtype=torch.float32).reshape(-1, 1)
        self.X = torch.tensor(frame.drop(columns=["target"]).to_numpy(), dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.network(x)

def evaluate(model, loader, criterion):
    model.eval(); total_loss=0; all_y=[]; all_pred=[]
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            all_y.extend(yb.cpu().numpy().ravel())
            all_pred.extend((torch.sigmoid(logits) >= 0.5).cpu().numpy().ravel())
    return total_loss / len(loader.dataset), accuracy_score(all_y, all_pred)

if __name__ == "__main__":
    data = load_breast_cancer()
    X_train, X_temp, y_train, y_temp = train_test_split(data.data, data.target, test_size=.30, stratify=data.target, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=.50, stratify=y_temp, random_state=SEED)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train); X_val = scaler.transform(X_val); X_test = scaler.transform(X_test)
    os.makedirs("artifacts", exist_ok=True)
    # Save a sample CSV and use the required CsvDataset class.
    csv_frame = pd.DataFrame(X_train, columns=data.feature_names); csv_frame["target"] = y_train
    csv_frame.to_csv("artifacts/breast_cancer_train.csv", index=False)
    train_ds = CsvDataset("artifacts/breast_cancer_train.csv")
    # validation/test use arrays so their labels never leak into training
    val_ds, test_ds = ArrayDataset(X_val, y_val), ArrayDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = MLP(X_train.shape[1]).to(DEVICE)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = copy.deepcopy(model.state_dict()); best_val = -1; patience=7; stale=0
    train_losses=[]
    for epoch in range(40):
        model.train(); running=0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
        train_losses.append(running / len(train_ds))
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1:02d}/40 train_loss={train_losses[-1]:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc; best_state = copy.deepcopy(model.state_dict()); stale=0
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping triggered."); break
    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), "artifacts/mlp_state_dict.pth")
    _, test_acc = evaluate(model, test_loader, criterion)
    # Reload into a fresh model and verify exactly equal test accuracy.
    reloaded = MLP(X_train.shape[1]).to(DEVICE)
    reloaded.load_state_dict(torch.load("artifacts/mlp_state_dict.pth", map_location=DEVICE, weights_only=True))
    reloaded.eval()
    _, reload_acc = evaluate(reloaded, test_loader, criterion)
    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Reloaded model accuracy: {reload_acc:.4f}; identical={abs(test_acc-reload_acc) < 1e-12}")
    lr = LogisticRegression(max_iter=2000, random_state=SEED).fit(X_train, y_train)
    print(f"Scikit-learn LogisticRegression test accuracy: {accuracy_score(y_test, lr.predict(X_test)):.4f}")
    pd.DataFrame({"epoch": range(1, len(train_losses)+1), "training_loss": train_losses}).to_csv("artifacts/training_loss.csv", index=False)
