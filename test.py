import numpy as np
from mpmath.libmp import normalize
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

x, y = make_moons(n_samples=1000, shuffle=True, noise=0.2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
norm = Normalizer()
norm.fit(x_train)
x_train_norm = norm.transform(x_train)
x_test_norm = norm.transform(x_test)
y_train_norm = norm.transform(y_train)
y_test_norm = norm.transform(y_test)
train_accuracy = accuracy_score(y_train_norm, x_train_norm)
test_accuracy = accuracy_score(y_test_norm, x_test_norm)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()
scaler = MinMaxScaler((-1, 1))
scaler.fit(x)
x_new = scaler.fit_transform(x)
plt.scatter(x_new[:, 0], x_new[:, 1], c=y)
plt.show()
