
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# ==========================================
# Load data
# ==========================================

data_path = Path(__file__).resolve().parent

train = pd.read_csv(data_path / "train.csv")
val = pd.read_csv(data_path / "val.csv")
test = pd.read_csv(data_path / "test.csv")

# Part (a): Visualize training data
plt.scatter(train["x1"], train["x2"], c=train["y"], cmap="viridis", s=10)
plt.colorbar(label="y")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Training Data")
plt.axis("equal")
plt.show()

def get_X(df):
    return torch.tensor(df[["x1", "x2"]].to_numpy(), dtype=torch.float32)

X_train = get_X(train)
X_val = get_X(val)
X_test = get_X(test)

y_train = torch.tensor(train["y"].to_numpy(), dtype=torch.float32)
y_val = torch.tensor(val["y"].to_numpy(), dtype=torch.float32)


def mse(predictions, y):
    return torch.mean((predictions - y)**2)


# Part (b)
# Sweep number of random feature pairs h

sigma = 1.0
lam = 0.001

h_values = [10, 18, 25, 50, 75, 100, 125, 200]

# Generate all random feature parameters once
max_h = max(h_values)

W = torch.randn(2, max_h, dtype=torch.float64) / sigma
b = 2 * torch.pi * torch.rand(max_h, dtype=torch.float64)

def rks_features(X, h):
    z = X.double() @ W[:, :h] + b[:h]
    return torch.cat([torch.sin(z), torch.cos(z)], dim=1)


# Ridge regression with an unpenalized intercept
def fit_ridge(Z, y, lam):
    y = y.double()

    Z_mean = Z.mean(dim=0)
    y_mean = y.mean()

    Z_centered = Z - Z_mean
    y_centered = y - y_mean

    D = Z.shape[1]
    I = torch.eye(D, dtype=torch.float64)

    weights = torch.linalg.solve(
        Z_centered.T @ Z_centered + len(Z) * lam * I,
        Z_centered.T @ y_centered
    )

    return weights, Z_mean, y_mean


def predict_ridge(Z, weights, Z_mean, y_mean):
    return (Z - Z_mean) @ weights + y_mean


# Hyperparameter sweep over h
rks_train_losses = []
rks_val_losses = []

for h in h_values:
    Z_train = rks_features(X_train, h)
    Z_val = rks_features(X_val, h)

    weights, Z_mean, y_mean = fit_ridge(Z_train, y_train, lam)

    train_pred = predict_ridge(Z_train, weights, Z_mean, y_mean)
    val_pred = predict_ridge(Z_val, weights, Z_mean, y_mean)

    train_loss = mse(train_pred, y_train).item()
    val_loss = mse(val_pred, y_val).item()

    rks_train_losses.append(train_loss)
    rks_val_losses.append(val_loss)

    print(f"h={h}: Train MSE={train_loss:.4f}, Val MSE={val_loss:.4f}")


# Plot hyperparameter sweep
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(h_values, rks_train_losses, marker="o")
axes[0].set_title("RKS Training MSE")
axes[0].set_xlabel("Number of random feature pairs (h)")
axes[0].set_ylabel("MSE")

axes[1].plot(h_values, rks_val_losses, marker="o")
axes[1].set_title("RKS Validation MSE")
axes[1].set_xlabel("Number of random feature pairs (h)")
axes[1].set_ylabel("MSE")

plt.tight_layout()
plt.show()


# Select best h using validation MSE
best_h = h_values[np.argmin(rks_val_losses)]

# Fit final RKS model
Z_train = rks_features(X_train, best_h)

rks_weights, rks_Z_mean, rks_y_mean = fit_ridge(
    Z_train, y_train, lam
)

print(f"Best RKS h: {best_h}")


def predict_rks(X):
    Z = rks_features(X, best_h)
    return predict_ridge(Z, rks_weights, rks_Z_mean, rks_y_mean)


# ==========================================
# Part (c): One-hidden-layer MLP
# ==========================================

# Architecture: R^2 -> R^width -> R
def make_mlp(width):
    return nn.Sequential(
        nn.Linear(2, width),
        nn.ReLU(),
        nn.Linear(width, 1)
    )


train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=256,
    shuffle=True
)


def train_mlp(width, epochs=100):
    torch.manual_seed(42)

    model = make_mlp(width)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()

        for X_batch, y_batch in train_loader:
            predictions = model(X_batch).squeeze(-1)
            loss = mse(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).squeeze(-1)
        val_pred = model(X_val).squeeze(-1)

        train_loss = mse(train_pred, y_train).item()
        val_loss = mse(val_pred, y_val).item()

    return model, train_loss, val_loss


# Sweep hidden layer width
widths = [8, 16, 32, 64, 128, 256, 512]

mlp_train_losses = []
mlp_val_losses = []

for width in widths:
    model, train_loss, val_loss = train_mlp(width)

    mlp_train_losses.append(train_loss)
    mlp_val_losses.append(val_loss)

    print(f"Width={width}: Train MSE={train_loss:.4f}, Val MSE={val_loss:.4f}")


# Plot hyperparameter sweep
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(widths, mlp_train_losses, marker="o")
axes[0].set_title("MLP Training MSE")
axes[0].set_xlabel("Hidden layer width")
axes[0].set_ylabel("MSE")

axes[1].plot(widths, mlp_val_losses, marker="o")
axes[1].set_title("MLP Validation MSE")
axes[1].set_xlabel("Hidden layer width")
axes[1].set_ylabel("MSE")

plt.tight_layout()
plt.show()


# Select best width using validation loss
best_width = widths[np.argmin(mlp_val_losses)]

# Refit final MLP
best_mlp, mlp_train_mse, mlp_val_mse = train_mlp(best_width)

print(f"Best MLP width: {best_width}")


def predict_mlp(X):
    best_mlp.eval()
    with torch.no_grad():
        return best_mlp(X).squeeze(-1)


# ==========================================
# Part (d): Compare models
# ==========================================

with torch.no_grad():
    rks_val_mse = mse(predict_rks(X_val), y_val).item()
    mlp_val_mse = mse(predict_mlp(X_val), y_val).item()

print("\nValidation MSE comparison:")
print(f"RKS: {rks_val_mse:.4f}")
print(f"MLP: {mlp_val_mse:.4f}")


# Predict on unlabeled test dataset
rks_test_predictions = predict_rks(X_test).numpy()
mlp_test_predictions = predict_mlp(X_test).numpy()

# One prediction per row, same order as test.csv
pd.DataFrame(rks_test_predictions).to_csv(
    data_path / "rks_test_predictions.csv",
    index=False, header=False
)

pd.DataFrame(mlp_test_predictions).to_csv(
    data_path / "mlp_test_predictions.csv",
    index=False, header=False
)


# ==========================================
# Visualize fitted functions
# ==========================================

# Grid over [-3, 3] x [-3, 3]
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)

grid_x1, grid_x2 = np.meshgrid(x1, x2)

grid = torch.tensor(
    np.column_stack([grid_x1.ravel(), grid_x2.ravel()]),
    dtype=torch.float32
)

rks_surface = predict_rks(grid).numpy().reshape(100, 100)
mlp_surface = predict_mlp(grid).numpy().reshape(100, 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

vmin = min(rks_surface.min(), mlp_surface.min())
vmax = max(rks_surface.max(), mlp_surface.max())

for ax, surface, title in zip(
    axes,
    [rks_surface, mlp_surface],
    ["RKS Fitted Function", "MLP Fitted Function"]
):
    plot = ax.contourf(
        grid_x1, grid_x2, surface,
        levels=30, vmin=vmin, vmax=vmax, cmap="viridis"
    )
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    fig.colorbar(plot, ax=ax)

plt.tight_layout()
plt.show()
