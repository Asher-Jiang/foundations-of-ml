import sklearn as sk
import sklearn.linear_model
import sklearn.svm
import scipy.stats as sp
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("ale_dataset.csv", delimiter=",", skip_header=1)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))

# Linear Regression residuals
reg_linear = sk.linear_model.LinearRegression().fit(data[:, 0:4], data[:, 4])
error_linear = data[:, 4] - reg_linear.predict(data[:, 0:4])
sp.probplot(error_linear, plot=axes[0,0])
normality = sp.normaltest(error_linear)
print("Normality of linear:",normality.pvalue)

# SVM residuals
reg_svm = sk.svm.SVR().fit(data[:, 0:4], data[:, 4])
error_svm = data[:, 4] - reg_svm.predict(data[:, 0:4])
sp.probplot(error_svm, plot=axes[0,1])
axes[0,1].set_title("SVM Error QQ Plot")
normality = sp.normaltest(error_svm)
print("Normality of svm:",normality.pvalue)

# Neural Network residuals
reg_nn = sk.neural_network.MLPRegressor(max_iter=1000).fit(data[:, 0:4], data[:, 4])
error_nn = data[:, 4] - reg_nn.predict(data[:, 0:4])
sp.probplot(error_nn, plot=axes[1,0])
axes[1,0].set_title("Neural Network Error QQ Plot")
normality = sp.normaltest(error_nn)
print("Normality of neural network:",normality.pvalue)

# Random forest residuals
reg_forest = sk.ensemble.RandomForestRegressor().fit(data[:, 0:4], data[:, 4])
error_forest = data[:, 4] - reg_forest.predict(data[:, 0:4])
sp.probplot(error_forest, plot=axes[1,1])
axes[1,1].set_title("Random Forest Error QQ Plot")
normality = sp.normaltest(error_forest)
print("Normality of random forest:",normality.pvalue)

plt.show()