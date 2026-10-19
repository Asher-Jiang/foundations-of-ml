import sklearn as sk
import sklearn.linear_model
import scipy.stats as sp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import StudentT
from pathlib import Path



def mean_absolute_error(y_true, y_pred):
    return sum(abs(y_true - y_pred)) / len(y_true)

def train_test_split(X, y, test_size=0.2):
    indices = np.random.permutation(len(X))
    index_split = int(len(X) * (1-test_size))

    train_idx = indices[:index_split]
    test_idx = indices[index_split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def sklearn_linreg(X_train, y_train, X_test, y_test):
    reg = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
    return mean_absolute_error(y_test, reg.predict(X_test))

# My implementation of linreg

def lin_model(xi, w, b):
    return (w @ xi.T + b)

def loss_lin(yi, fxi):
    return ((yi - fxi) ** 2)

def train_linreg(dataset, step_count=200):
    w = torch.zeros(4, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([w,b], lr=0.005)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i in range(step_count):
        for X, y in loader:
            optimizer.zero_grad()
            L = loss_lin(y, lin_model(X, w, b)).mean()

            L.backward()
            optimizer.step()

    return w, b

def predict_linreg(w, b, X_test):
    return (lin_model(X_test, w, b))

def eval_accuracy(prediction, y_test):
    return (abs((prediction - y_test)).mean())

# T-distribution

def loss_t(yi, fxi, v=5):
    return -StudentT(df=v).log_prob(yi - fxi)

def train_t_dist(dataset, step_count=200):
    w = torch.zeros(4, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([w,b], lr=0.005)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i in range(step_count):
        for X, y in loader:
            optimizer.zero_grad()
            L = loss_t(y, lin_model(X, w, b)).mean()

            L.backward()
            optimizer.step()

    return w, b


if __name__=="__main__":

    # Load data and 80-20 split
    data = torch.tensor(np.genfromtxt(Path(__file__).resolve().parent / "ale_dataset.csv", delimiter=",", skip_header=1), dtype=torch.float32)
    X, y = data[:, 0:4], data[:, 4]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # sklearn linreg
    print(f"sklearn MAE: {sklearn_linreg(X_train, y_train, X_test, y_test)}")

    # My linreg

    mu, sd = X_train.mean(0), X_train.std(0)
    X_train_s = (X_train - mu) / sd
    X_test_s  = (X_test  - mu) / sd
    dataset = TensorDataset(X_train_s, y_train)
    w, b = train_linreg(dataset)
    predictions = predict_linreg(w, b, X_test_s)
    mae_linreg = eval_accuracy(predictions, y_test)
    print(f"MAE of my linreg model: {mae_linreg}")

    # My t-dist regression
    dataset = TensorDataset(X_train_s, y_train)
    w_t, b_t = train_t_dist(dataset)
    predictions = predict_linreg(w_t, b_t, X_test_s)
    mae_linreg = eval_accuracy(predictions, y_test)
    print(f"MAE of my t-dist regression model: {mae_linreg}")




