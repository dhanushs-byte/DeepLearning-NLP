"""Task 1: NumPy XOR network, activations/losses, autograd, and learning-rate experiment."""
import numpy as np
import matplotlib.pyplot as plt
import torch

np.random.seed(42)
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([[0.], [1.], [1.], [0.]])

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    z = np.asarray(x, dtype=float) - np.max(x)
    e = np.exp(z)
    return e / e.sum()

def mse(y_true, y_pred):
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)

def cross_entropy(y_true, y_pred):
    p = np.clip(np.asarray(y_pred), 1e-7, 1 - 1e-7)
    y_true = np.asarray(y_true)
    return -np.mean(y_true * np.log(p) + (1-y_true) * np.log(1-p))

def train_xor(lr=0.5, epochs=5000, seed=42):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, (2, 4)); b1 = np.zeros((1, 4))
    W2 = rng.normal(0, 1, (4, 1)); b2 = np.zeros((1, 1))
    losses = []
    for epoch in range(epochs):
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        loss = cross_entropy(y, a2)
        losses.append(loss)
        # BCE + sigmoid gives dL/dz2 = (prediction - target) / number of rows
        dz2 = (a2 - y) / len(X)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        with np.errstate(over="ignore", invalid="ignore"):
            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2
        if not np.isfinite(loss) or not all(np.isfinite(a).all() for a in (W1,b1,W2,b2)):
            losses.extend([np.nan] * (epochs - len(losses)))
            break
    pred = sigmoid(relu(X @ W1 + b1) @ W2 + b2)
    return np.array(losses), pred

if __name__ == "__main__":
    print("softmax([2.0, 1.0, 0.1]) =", np.round(softmax([2.0, 1.0, 0.1]), 3))
    # Hand calculation: L=(wx+b-t)^2; dL/dw=2(wx+b-t)x; dL/db=2(wx+b-t)
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    x = torch.tensor(2.0); target = torch.tensor(10.0)
    loss = (w*x + b - target) ** 2
    loss.backward()
    print("PyTorch autograd: w.grad =", w.grad.item(), "b.grad =", b.grad.item())
    print("Hand gradients: dw = -32, db = -16")
    for lr in (0.01, 0.5, 5.0):
        losses, predictions = train_xor(lr=lr)
        print(f"lr={lr}: final loss={losses[np.isfinite(losses)][-1] if np.isfinite(losses).any() else 'diverged'}")
        if lr == 0.5:
            print("XOR predictions:", np.round(predictions.ravel(), 2))
            plt.figure()
            plt.plot(losses)
            plt.title("XOR Training Loss (learning rate 0.5)")
            plt.xlabel("Epoch"); plt.ylabel("Binary cross-entropy")
            plt.tight_layout(); plt.savefig("task1_loss.png", dpi=150)
            plt.show()
    print("Dropout: Randomly turns off some activations during training to reduce overfitting.")
    print("BatchNorm: Normalizes layer activations within a mini-batch to stabilize training.")
    print("Early stopping: Stops training when validation performance stops improving.")
