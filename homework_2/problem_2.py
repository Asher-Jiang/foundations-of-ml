
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from category_encoders import TargetEncoder
from torch.utils.data import random_split, TensorDataset, DataLoader

torch.manual_seed(42)


# Part (a): Load and split data


data_path = Path(__file__).resolve().parent / "retail.csv"
data = pd.read_csv(data_path)

categorical_columns = ["StockCode", "CustomerID"]

for column in categorical_columns:
    data[column] = data[column].astype("category")

print(data[categorical_columns].nunique())

train_rows, val_rows, test_rows = random_split(
    range(len(data)),
    [0.7, 0.15, 0.15],
    generator=torch.Generator().manual_seed(42)
)

train_data = data.iloc[train_rows.indices]
validation_data = data.iloc[val_rows.indices]
test_data = data.iloc[test_rows.indices]

y_train = torch.tensor(train_data["Quantity"].to_numpy(), dtype=torch.float32)
y_val = torch.tensor(validation_data["Quantity"].to_numpy(), dtype=torch.float32)
y_test = torch.tensor(test_data["Quantity"].to_numpy(), dtype=torch.float32)

batch_size = 256



# Shared training and evaluation functions


# Predict log(1 + Quantity) using squared error
def loss_function(predictions, y):
    return torch.mean((predictions - torch.log1p(y))**2)


def evaluate(forward_fn, data_loader):
    total_mse = 0
    total_mae = 0

    with torch.no_grad():
        for batch in data_loader:
            *inputs, y = batch

            predictions = forward_fn(*inputs)

            mse = loss_function(predictions, y)
            quantities = torch.expm1(predictions)
            mae = torch.abs(quantities - y).mean()

            total_mse += mse.item() * len(y)
            total_mae += mae.item() * len(y)

    N = len(data_loader.dataset)
    return total_mse / N, total_mae / N


def train_model(forward_fn, train_loader, val_loader, optimizer, epochs=30):
    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            *inputs, y = batch

            predictions = forward_fn(*inputs)
            loss = loss_function(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, _ = evaluate(forward_fn, val_loader)

        print(
            f"Epoch {epoch+1}: "
            f"Train MSE = {train_loss:.4f}, "
            f"Val MSE = {val_loss:.4f}"
        )



# Part (b): Target encoding + linear model


target_encoder = TargetEncoder(cols=categorical_columns)

X_train = target_encoder.fit_transform(
    train_data[categorical_columns], train_data["Quantity"]
)
X_val = target_encoder.transform(validation_data[categorical_columns])
X_test = target_encoder.transform(test_data[categorical_columns])

X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)

# Standardize target-encoded features using training statistics only
# IMPORTANT: Do this BEFORE creating DataLoaders
mean = X_train.mean(dim=0)
std = X_train.std(dim=0).clamp(min=1e-6)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

print("Standardized feature means:", X_train.mean(dim=0))
print("Standardized feature std:", X_train.std(dim=0))

# Check that inputs and targets are finite
assert torch.isfinite(X_train).all()
assert torch.isfinite(X_val).all()
assert torch.isfinite(X_test).all()
assert torch.isfinite(y_train).all()
assert torch.isfinite(y_val).all()
assert torch.isfinite(y_test).all()

train_loader_b = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size, shuffle=True
)
val_loader_b = DataLoader(
    TensorDataset(X_val, y_val), batch_size=batch_size
)
test_loader_b = DataLoader(
    TensorDataset(X_test, y_test), batch_size=batch_size
)

linear_model = nn.Linear(2, 1)

def forward_b(X):
    return linear_model(X).squeeze(-1)

optimizer_b = torch.optim.SGD(
    linear_model.parameters(), lr=0.001
)

print("\nPart (b): Target encoding")
train_model(forward_b, train_loader_b, val_loader_b, optimizer_b)



# Part (c): Embeddings + MLP


# Assign integer indices using training data only
stock_to_idx = {
    code: i + 1
    for i, code in enumerate(train_data["StockCode"].unique())
}

customer_to_idx = {
    customer: i + 1
    for i, customer in enumerate(train_data["CustomerID"].unique())
}

# Index 0 represents unseen categories
def encode_categories(df):
    stock = df["StockCode"].astype("object").map(stock_to_idx).fillna(0)
    customer = df["CustomerID"].astype("object").map(customer_to_idx).fillna(0)

    return torch.tensor(
        list(zip(stock, customer)), dtype=torch.long
    )


X_train_emb = encode_categories(train_data)
X_val_emb = encode_categories(validation_data)
X_test_emb = encode_categories(test_data)

embedding_dim = 8
width = 16

num_stocks = len(stock_to_idx) + 1
num_customers = len(customer_to_idx) + 1

stock_embedding = nn.Embedding(num_stocks, embedding_dim)
customer_embedding = nn.Embedding(num_customers, embedding_dim)

mlp = nn.Sequential(
    nn.Linear(2 * embedding_dim, width),
    nn.ReLU(),
    nn.Linear(width, width),
    nn.ReLU(),
    nn.Linear(width, 1)
)

def forward_c(X):
    stock_vectors = stock_embedding(X[:, 0])
    customer_vectors = customer_embedding(X[:, 1])

    X_embedded = torch.cat(
        [stock_vectors, customer_vectors], dim=1
    )

    return mlp(X_embedded).squeeze(-1)


train_loader_c = DataLoader(
    TensorDataset(X_train_emb, y_train),
    batch_size=batch_size, shuffle=True
)
val_loader_c = DataLoader(
    TensorDataset(X_val_emb, y_val), batch_size=batch_size
)
test_loader_c = DataLoader(
    TensorDataset(X_test_emb, y_test), batch_size=batch_size
)

optimizer_c = torch.optim.SGD(
    list(stock_embedding.parameters())
    + list(customer_embedding.parameters())
    + list(mlp.parameters()),
    lr=0.001
)

print("\nPart (c): Embeddings + MLP")
train_model(forward_c, train_loader_c, val_loader_c, optimizer_c)


# Part (d): Target encoding + embeddings + MLP

# Separate embeddings and MLP for this model
stock_embedding_d = nn.Embedding(num_stocks, embedding_dim)
customer_embedding_d = nn.Embedding(num_customers, embedding_dim)

combined_mlp = nn.Sequential(
    nn.Linear(2 * embedding_dim + 2, width),
    nn.ReLU(),
    nn.Linear(width, width),
    nn.ReLU(),
    nn.Linear(width, 1)
)

def forward_d(X_emb, X_target):
    stock_vectors = stock_embedding_d(X_emb[:, 0])
    customer_vectors = customer_embedding_d(X_emb[:, 1])

    X_combined = torch.cat(
        [stock_vectors, customer_vectors, X_target], dim=1
    )

    return combined_mlp(X_combined).squeeze(-1)


# X_train, X_val, X_test are already standardized
train_loader_d = DataLoader(
    TensorDataset(X_train_emb, X_train, y_train),
    batch_size=batch_size, shuffle=True
)
val_loader_d = DataLoader(
    TensorDataset(X_val_emb, X_val, y_val),
    batch_size=batch_size
)
test_loader_d = DataLoader(
    TensorDataset(X_test_emb, X_test, y_test),
    batch_size=batch_size
)

optimizer_d = torch.optim.SGD(
    list(stock_embedding_d.parameters())
    + list(customer_embedding_d.parameters())
    + list(combined_mlp.parameters()),
    lr=0.001
)

print("\nPart (d): Combined model")
train_model(forward_d, train_loader_d, val_loader_d, optimizer_d)


# Final comparison on test set

mse_b, mae_b = evaluate(forward_b, test_loader_b)
mse_c, mae_c = evaluate(forward_c, test_loader_c)
mse_d, mae_d = evaluate(forward_d, test_loader_d)

print("\nTest set results:")
print("Model                 Log-MSE      MAE")
print(f"Target encoding       {mse_b:.4f}       {mae_b:.4f}")
print(f"Embeddings + MLP      {mse_c:.4f}       {mae_c:.4f}")
print(f"Combined model        {mse_d:.4f}       {mae_d:.4f}")
