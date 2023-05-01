# 最终在main函数中传入一个维度为6的numpy数组，输出预测值
# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    x, y = read_data()
    a = 0.5
    E = np.eye(np.linalg.inv(np.dot(x.T, x)))
    return np.dot(np.linalg.inv(np.dot(x, x.T) + np.dot(a, E)), np.dot(x, y))


def lasso(data):
    x, y = read_data()
    w = np.random.rand(1, 6)
    a = 0.5
    p = np.dot(x.T, y - np.dot(x.T, w))
    z = np.dot(x.T, x)
    if p < (-0.5 * a):
        return (p + 0.5 * a) / z
    elif p > (0.5 * a):
        return (p - 0.5 * a) / z
    else:
        return 0


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

