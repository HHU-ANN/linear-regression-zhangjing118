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
    weight = model1(x, y)
    return data @ weight


def model1(x, y):
    E=np.eye(6)
    a=0.5
    return np.dot(np.linalg.inv(np.dot(x.T, x)+a*E), np.dot(x.T, y))


def lasso(data):
    x, y = read_data()
    weight=model2(x,y,10,0.5)
    return data @ weight


def model2(X, y, iternum, lamda):
    m, n = X.shape
    theta = np.matrix(np.zeros((n, 1)))
    # 循环
    for it in range(iternum):
        for k in range(n):  # n个特征
            # 计算z_k和p_k
            z_k = np.sum(np.power(X[:, k], 2))
            p_k = 0
            for i in range(m):
                p_k += X[i, k] * (y[i] - np.sum([X[i, j] * theta[j] for j in range(n) if j != k]))
            # 根据p_k的不同取值进行计算
            if p_k < -lamda / 2:
                w_k = (p_k + lamda / 2) / z_k
            elif p_k > lamda / 2:
                w_k = (p_k - lamda / 2) / z_k
            else:
                w_k = 0
            theta[k] = w_k
    return theta
    
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

