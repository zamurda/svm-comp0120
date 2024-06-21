from svm.svm import SVM
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

rng = default_rng(53)

# generate radially separable dataset
# adapted from https://s3.amazonaws.com/assets.datacamp.com/production/course_6651/slides/chapter3.pdf
n = 1000
x1 = rng.uniform(-1, 1, n)
x2 = rng.uniform(-1, 1, n)

r = 0.6
y = np.where(x1**2 + x2**2 < r**2, 1, -1)
X = np.column_stack((x1, x2))

_, unique_indices = np.unique(X, return_index=True, axis=0)
X = X[unique_indices]
y = y[unique_indices]

## train SVM
my_svm = SVM(C=1.0, kernel='rbf', max_iter=1000)
clf = my_svm.fit(X,y)

## plot result
kernel = my_svm.K
b = my_svm.b
alpha = my_svm.dual_variables
u = lambda xi: np.sum(alpha[alpha != 0]*y[alpha != 0]*np.array([kernel(xi, j) for j in X[alpha != 0]])) - b

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - .1, x.max() + .1
    y_min, y_max = y.min() - .1, y.max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, xx, yy, **params):
    points = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    for p in points:
        if u(p) >= 0: Z.append(1)
        else: Z.append(-1)
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('$x_2$')
ax.set_xlabel('$x_1$')
ax.set_title('Classification Result')