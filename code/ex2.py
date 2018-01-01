# -*- coding: utf-8 -*-
import numpy as np
import math

"""
Sun. Dec 12th 2017, 11:36

@author: Yuzhao Chen
"""

def initTime(xs, problem=1):
    if problem == 1:
        return [2 * np.sin(2 * math.pi * x) for x in xs]
    elif problem == 2:
        return [np.sin(math.pi * x) + x * (1 - x) for x in xs]
    else:
        raise ValueError("Invalid Input.")

def initBorder(t, problem=1):
    if problem == 1:
        return 0
    elif problem == 2:
        return 0
    else:
        raise ValueError("Invalid Input.")

def precisionSolution(x,t,problem=1):
    if problem==1:
        return 2 * math.e**(- math.pi**2 / 4 * t) * np.sin(2 * math.pi * x)
    elif problem == 2:
        return math.e**(-t * math.pi**2) * np.sin(math.pi * x) + x * (1 - x)
    else:
        raise ValueError("Invalid Input.")


def forward(M, N, ex=1):
    if ex==1:
        a = 1 / 16
        Fn = np.zeros((N, M - 1))  # 题目特殊，Fn都是常数
        x1 = 0
        x2 = 1
        t1 = 0
        t2 = 1
    elif ex==2:
        a = 1
        Fn = np.ones((N, M - 1)) * 2 # 题目特殊，Fn都是常数
        x1 = 0
        x2 = 1
        t1 = 0
        t2 = 2

    h = (x2 - x1) / M  # 空间步长(横轴）
    k = (t2 - t1) / N  # 时间步长（纵轴）
    print("空间步长：", h, end=',')
    print("时间步长：", k)
    r = a * k / (h**2) # 网比
    print("r值（＜1/2时稳定）：", r)

    xs = np.arange(x1, x2 + h, h)  # 空间节点序列
    ts = np.arange(t1, t2 + k, k)  # 时间节点序列

    U = np.zeros((N, M-1)) # 不同行不同时间，不同列不同空间

    U[0][0] = U[0][-1] = 0
    U[0] = initTime(xs[1:-1], problem=ex)

    I = np.eye(M-1)
    C = np.eye(M-1, k=1) + np.eye(M-1, k=-1)

    for i in range(1, U.shape[0]):
        U[i] = np.dot((1 - 2*r) * I + r * C, U[i-1]) + k * Fn[i-1]

    UTrue = np.zeros((N, M-1))
    UTrue[0] = U[0]
    i = 0
    for t in ts[1:-1]:
        i += 1
        j = -1
        for x in xs[1:-1]:
            j += 1
            UTrue[i][j] = precisionSolution(x,t, problem=ex)

    U_array = U.reshape(1, N * (M-1))
    UTrue_array = UTrue.reshape(1, N * (M-1))

    return U, UTrue, np.linalg.norm(U_array - UTrue_array)

def backward(M, N, ex=1):
    if ex==1:
        a = 1 / 16
        Fn = np.zeros((N, M - 1))  # 题目特殊，Fn都是常数
        x1 = 0
        x2 = 1
        t1 = 0
        t2 = 1
    elif ex==2:
        a = 1
        Fn = np.ones((N, M - 1)) * 2 # 题目特殊，Fn都是常数
        x1 = 0
        x2 = 1
        t1 = 0
        t2 = 2

    h = (x2 - x1) / M  # 空间步长(横轴）
    k = (t2 - t1) / N  # 时间步长（纵轴）
    print("空间步长：", h, end=',')
    print("时间步长：", k)
    r = a * k / (h**2) # 网比
    print("r值（绝对稳定）：", r)

    xs = np.arange(x1, x2 + h, h)  # 空间节点序列
    ts = np.arange(t1, t2 + k, k)  # 时间节点序列

    U = np.zeros((N, M-1)) # 不同行不同时间，不同列不同空间

    U[0] = initTime(xs[1:-1], problem=ex)

    I = np.eye(M-1)
    C = np.eye(M-1, k=1) + np.eye(M-1, k=-1)

    for i in range(1, U.shape[0]):
        U[i] = np.linalg.solve((1 + 2*r)*I - r*C, U[i-1] + k*Fn[i])

    UTrue = np.zeros((N, M-1))
    UTrue[0] = U[0]
    i = 0
    for t in ts[1:-1]:
        i += 1
        j = -1
        for x in xs[1:-1]:
            j += 1
            UTrue[i][j] = precisionSolution(x,t, problem=ex)

    U_array = U.reshape(1, N * (M-1))
    UTrue_array = UTrue.reshape(1, N * (M-1))

    return U, UTrue, np.linalg.norm(U_array - UTrue_array)

def sixPoint(M, N, ex=1):
    if ex==1:
        a = 1 / 16
        Fn = np.zeros((N, M - 1))  # 题目特殊，Fn都是常数
        x1 = 0
        x2 = 1
        t1 = 0
        t2 = 1
    elif ex==2:
        a = 1
        Fn = np.ones((N, M - 1)) * 2 # 题目特殊，Fn都是常数
        x1 = 0
        x2 = 1
        t1 = 0
        t2 = 2

    h = (x2 - x1) / M  # 空间步长(横轴）
    k = (t2 - t1) / N  # 时间步长（纵轴）
    print("空间步长：", h, end=',')
    print("时间步长：", k)
    r = a * k / (h**2) # 网比
    print("r值（绝对稳定）：", r)

    xs = np.arange(x1, x2 + h, h)  # 空间节点序列
    ts = np.arange(t1, t2 + k, k)  # 时间节点序列

    U = np.zeros((N, M-1)) # 不同行不同时间，不同列不同空间

    U[0] = initTime(xs[1:-1], problem=ex)

    I = np.eye(M-1)
    C = np.eye(M-1, k=1) + np.eye(M-1, k=-1)

    for i in range(1, U.shape[0]):
        U[i] = np.linalg.solve((1 + r)*I - 0.5*r*C, np.dot((1-r)*I+0.5*r*C,U[i-1]) + 0.5*k*(Fn[i]+Fn[i-1]))

    UTrue = np.zeros((N, M-1))
    UTrue[0] = U[0]
    i = 0
    for t in ts[1:-1]:
        i += 1
        j = -1
        for x in xs[1:-1]:
            j += 1
            UTrue[i][j] = precisionSolution(x,t, problem=ex)

    U_array = U.reshape(1, N * (M-1))
    UTrue_array = UTrue.reshape(1, N * (M-1))

    return U, UTrue, np.linalg.norm(U_array - UTrue_array)

# [4,4],[8,16],[8,8],[16,32],[32, 64]
for x, y in [[4, 100],[8,400],[4,200],[8, 800],[8, 200]]:
    print("空间节点数量: ", x, end=',')
    print("时间节点数量: ", y)
    result = sixPoint(x,y, ex=2)

    print("向量差L2范数（误差）：",result[2])
    print()





