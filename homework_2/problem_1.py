import torch
from tensordict import TensorDictBase
import pandas as pd
import torch.nn as nn
from torch.utils.data import random_split, TensorDataset, DataLoader

width = 16
n = 3
data = TensorDictBase.from_parquet("homework_2/taxi_trips.parquet")

feature_names = [
    name for name in data.keys()
    if name not in ["trip_duration", "__index_level_0__"]
]

X = torch.stack(
    [data[name] for name in feature_names],
    dim=1
).float()

y = data["trip_duration"].float().reshape(-1)

# Combine features and targets into one dataset
dataset = TensorDataset(X, y)

# test/training/val split
train_set, val_set, test_set = random_split(
    dataset,
    [0.7, 0.15, 0.15],
    generator=torch.Generator().manual_seed(42)
)

# Create DataLoaders
batch_size = 256

train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

val_loader = DataLoader(
    val_set, batch_size=batch_size
)

test_loader = DataLoader(
    test_set, batch_size=batch_size
)

# Initalize weights
layers = []

for i in range(n):
    if i == 0:
        layers.append(nn.Linear(7, width))
    elif i == n - 1:
        layers.append(nn.Linear(width, 2))
    else:
        layers.append(nn.Linear(width, width))

    if i != n - 1:
        layers.append(nn.ReLU())

model = nn.Sequential(*layers)

def loss_function(y, z):
    # link function
    mu = z[:, 0]
    sigma = torch.exp(z[:, 1])

    dist = torch.distributions.Normal(mu, sigma)
    return -dist.log_prob(y).mean()

# Training loop:

# Vanilla SGD
step_size = 0.005
optimizer = torch.optim.SGD(model.parameters(), lr=step_size)

loops = 50

for loop in range(loops):
    total_loss = 0

    for X_batch, y_batch in train_loader:
        # Forward pass
        z = model(X_batch)

        # Calculate loss
        loss = loss_function(y_batch, z)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    avg_loss = total_loss / len(train_set)
    print(f"loop {loop + 1}: loss = {avg_loss:.4f}")


# Baseline model: linear regression over features with uniform (learned) standard deviation
linear_model = nn.Linear(7, 1)
log_sigma = nn.Parameter(torch.tensor(0.0))

optimizer = torch.optim.SGD(
    list(linear_model.parameters()) + [log_sigma],
    lr=0.001
)

def linear_baseline_loss(X, y):
    mu = linear_model(X).squeeze(-1)
    sigma = torch.exp(log_sigma)

    dist = torch.distributions.Normal(mu, sigma)
    return -dist.log_prob(y).mean()


# Train linear gaussian model
baseline_loops = 50

for loops in range(baseline_loops):
    total_loss = 0

    for X_batch, y_batch in train_loader:
        # Calculate baseline loss
        loss = linear_baseline_loss(X_batch, y_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    avg_loss = total_loss / len(train_set)
    print(f"Baseline loop {loops+1}: loss = {avg_loss:.4f}")


# Evaluation of DNN vs. Linear gaussian model:

model.eval()
linear_model.eval()

mlp_test_loss = 0
baseline_test_loss = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        # DNN
        z = model(X_batch)
        mlp_test_loss += loss_function(y_batch, z).item() * len(y_batch)

        # Linear Gaussian baseline
        baseline_test_loss += (
            linear_baseline_loss(X_batch, y_batch).item() * len(y_batch)
        )

mlp_test_loss /= len(test_set)
baseline_test_loss /= len(test_set)

print(f"MLP test NLL: {mlp_test_loss:.4f}")
print(f"Linear Gaussian test NLL: {baseline_test_loss:.4f}")


# Part (e): Evaluate probability predictions on test set

model.eval()

X_test = X[test_set.indices]
y_test = y[test_set.indices]

with torch.no_grad():
    # Predict Gaussian parameters
    z = model(X_test)
    mu = z[:, 0]
    sigma = torch.exp(z[:, 1])

    # Probability that trip takes less than 45 minutes
    dist = torch.distributions.Normal(mu, sigma)
    p = dist.cdf(torch.tensor(45.0))

    # Actual binary outcomes
    t = (y_test < 45).float()

    # Binary log-loss
    eps = 1e-7
    p = p.clamp(eps, 1 - eps)

    log_loss = -(
        t * torch.log(p) + (1 - t) * torch.log(1 - p)
    ).mean()

    # Classification accuracy
    predictions = (p >= 0.5).float()
    accuracy = (predictions == t).float().mean()

print(f"Test log-loss: {log_loss.item():.4f}")
print(f"Test accuracy: {accuracy.item():.4f}")

