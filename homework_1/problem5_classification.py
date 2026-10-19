import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import math
import torch
from torch.utils.data import TensorDataset, DataLoader

def train_test_split(X, y, test_size=0.2):
    indices = np.random.permutation(len(X))
    index_split = int(len(X) * (1-test_size))

    train_idx = indices[:index_split]
    test_idx = indices[index_split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def scikit_log_reg(X, y):
    clf = LogisticRegression(random_state=0).fit(X, y)

    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    return accuracy

def sigmoid(x):
    return torch.clamp(1 / (1 + torch.exp(-x)), min=1e-7, max=1-1e-7)

def log_reg(xi, w, b):
    return sigmoid(xi @ w + b)

def loss(yi, fxi):
    return(-(yi * torch.log(fxi) + (1-yi) * torch.log(1-fxi)))


def train_logistic(dataset, step_count=300):
    w = torch.zeros(30, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([w,b], lr=0.05)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i in range(step_count):
        for X, y in loader:
            optimizer.zero_grad()
            L = loss(y, log_reg(X, w, b)).mean()

            L.backward()
            optimizer.step()

    return w, b


def probit_reg(xi, w, b):
    normal_dist = torch.distributions.Normal(0, 1)
    return torch.clamp(normal_dist.cdf(xi @ w + b), min=1e-7, max=1-1e-7)

def train_probit(dataset, step_count=300):
    w = torch.zeros(30, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([w,b], lr=0.05)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i in range(step_count):
        for X, y in loader:
            optimizer.zero_grad()
            L = loss(y, probit_reg(X, w, b)).mean()

            L.backward()
            optimizer.step()

    return w, b


def predict_log(w, b, X_test):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    return(log_reg(X_test_tensor, w , b) > 0.5)

def predict_probit(w, b, X_test):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    return(probit_reg(X_test_tensor, w , b) > 0.5)
    
def eval_accuracy(prediction, y_test):
    correct = 0
    for i in range(len(prediction)):
        if prediction[i] == y_test[i]:
            correct += 1
    return (correct / len(prediction))


if __name__=="__main__":

    data = sklearn.datasets.load_breast_cancer(return_X_y=False, as_frame=False)

    # 80-20 training-test split
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # sklearn log reg implementation.
    print(f"Accuracy of sklearn model: {scikit_log_reg(X_train, y_train)}")

    # My own implementation of logistic
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    w_log, b_log = train_logistic(dataset)
    predictions_log = predict_log(w_log, b_log, X_test)
    accuracy_log = eval_accuracy(predictions_log, y_test)
    print(f"Accuracy of my logistic model: {accuracy_log}")

    # My implementation of probit
    w_prob, b_prob = train_probit(dataset)
    predictions_probit = predict_probit(w_prob, b_prob, X_test)
    accuracy_probit = eval_accuracy(predictions_probit, y_test)
    print(f"Accuracy of my probit model: {accuracy_probit}")





