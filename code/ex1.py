import numpy as np
import math

"""
author: Yuzhao Chen, 2017.10.26
"""

'''
-d2u / dx2 - d2u / dy2 = f(x,y);
1. f(x,y) = 0, 1 < x < 2, 0 < y < 1; 
    u(x, 0) = 2lnx, u(x, 1) = ln(x^2 + 1) for 1<x<2;
    u(1, y) = ln(1 + y^2), u(2, y) = ln(4 + y^2) for 0<y<1;
Precision Solution is: u(x, y) = ln(x^2 + y^2).

2. f(x, y) = -4, 0 < x < 1, 0 < y < 2;
    u(x, 0) = x^2, u(x, 2) = (x-2)^2, for 0<x<1;
    u(0, y) = y^2, u(1, y) = (y - 1)^2 for 0<y<2;

3. f(x,y) = cos(x + y) + cos(x - y)
    pass tempolarily.
'''
def f(x,y, problem=1):
    if problem == 1:
        return 0
    elif problem == 2:
        return -4
    elif problem == 3:
        return np.cos(x+y) + np.cos(x-y)
    else:
        raise ValueError("Input Invalid!")

def bottom(x, problem=1):
    # u(x,0)
    if problem == 1:
        return 2 * math.log(x, math.e)
    elif problem == 2:
        return x**2
    elif problem == 3:
        return np.cos(x)
    else:
        raise ValueError("Input Invalid!")

def top(x, problem=1):
    # u(x, upperBound)
    if problem == 1:
        return math.log(x**2 + 1, math.e)
    elif problem == 2:
        return (x - 2)**2
    elif problem == 3:
        return 0
    else:
        raise ValueError("Input Invalid")

def left(y, problem=1):
    # u(1, y) or u(0, y) depend on diff problem
    if problem == 1:
        return math.log((1+y**2), math.e)
    elif problem == 2:
        return y**2
    elif problem == 3:
        return np.cos(y)
    else:
        raise ValueError("Input Invalid!")

def right(y, problem=1):
    if problem == 1:
        return math.log((4+y**2), math.e)
    elif problem == 2:
        return (y-1)**2
    elif problem == 3:
        return -np.cos(y)
    else:
        raise ValueError("Input Invalid!")

def precisionSolution(x,y, problem=1):
    if problem == 1:
        return math.log((x**2 + y**2), math.e)
    elif problem == 2:
        return (x-y)**2
    elif problem == 3:
        return np.cos(x) * np.cos(y)
    else:
        raise ValueError("Input Invalid!")

def fivePoints(x1, x2, y1, y2, M, N, ex=1):
    h = (x2 - x1) / M  # 空间步长(横轴）
    k = (y2 - y1) / N  # 时间步长（纵轴）

    xs = np.arange(x1, x2 + h, h) # 空间节点序列
    ys = np.arange(y1, y2 + k, k) # 时间节点序列

    # 转化为求解线性方程组AUn=Fn
    # 先定义五点格式方程系数
    p = 1  # 泊松方程p(x,y)=1, q(x,y)=0
    a1 = 1/(h**2)
    a2 = 1/(k**2)
    a3 = 1/(h**2)
    a4 = 1/(k**2)
    a0 = a1+a2+a3+a4
    print("a4", a4)
    # 定义矩阵A的对角元素B
    B = np.zeros((M - 1, M - 1))
    for i in range(B.shape[0]):
        B[i,i] = a0
        if i == 0:
            B[i, i+1] = -a1
        elif i == M - 2:
            B[i, i-1] = -a3
        else:
            B[i, i-1] = -a3
            B[i, i+1] = -a1

    # 定义构成A的辅助M-1阶单位矩阵
    I = np.eye(M-1)

    # 定义矩阵A
    A = np.zeros(((N-1)*(M-1), (N-1)*(M-1)))
    for i in range(N-1):

            A[i*(M-1):i*(M-1)+(M-1),i*(M-1):i*(M-1)+(M-1)] = B
            if i == 0:
                A[i*(M-1):i*(M-1)+(M-1), (i+1)*(M-1):(i+1)*(M-1)+(M-1)] = -a2 * I
            elif i == N-2:
                A[i*(M-1):i*(M-1)+(M-1), (i-1)*(M-1):(i-1)*(M-1)+(M-1)] = -a4 * I
            else:
                A[i*(M-1):i*(M-1)+(M-1), (i-1)*(M-1):(i-1)*(M-1)+(M-1)] = -a4 * I
                A[i*(M-1):i*(M-1)+(M-1), (i+1)*(M-1):(i+1)*(M-1)+(M-1)] = -a2 * I

    # 最后定义向量Fn
    Fn = [] # 注意从下到上，从左到右的顺序
    for y in ys[1:-1]:
        for x in xs[1:-1]:
            Fn.append(f(x, y, problem=ex))
    Fn = np.array(Fn).astype(float)

    # 还要处理边缘条件
    Fn = Fn.reshape(N-1, M-1)
    for i in range(Fn.shape[1]):
        Fn[0, i] += a4 * bottom(xs[i+1], problem=ex) # xi, i = 1,2,...,M-1, not use i=0andM
        Fn[N-1-1, i] += a2 * top(xs[i+1], problem=ex)
    for i in range(Fn.shape[0]):
        Fn[i, 0] += a3 * left(ys[i+1], problem=ex)
        Fn[i, M-1-1] += a1 * right(ys[i+1], problem=ex)

    Fn = Fn.reshape(1,(N-1)*(M-1))
    Un = np.linalg.solve(A, Fn.T)
    Un = Un.reshape(1,(N-1)*(M-1))

    uTrue = []
    for y in ys[1:-1]:
        for x in xs[1:-1]:
            uTrue.append(precisionSolution(x, y, problem=ex))
    uTrue = np.array(uTrue)

    return h, k, np.linalg.norm(Un-uTrue)

if __name__ == '__main__':
    # ex1
    print("for exercise one:")
    for M in [4, 8, 16, 36]:
        U1 = fivePoints(1, 2, 0, 1, M, M, ex=1)
        print()
        print("x轴步长：", U1[0])
        print("y轴步长：", U1[1])
        print("误差：", U1[2])
        break
    '''
    print("for exercise two:")

    for M in [4, 8, 16, 36, 64]:
        U2 = fivePoints(0, 1, 0, 2,M,M, ex=2)
        print()
        print("x轴步长：", U2[0])
        print("y轴步长：", U2[1])
        print("误差：", U2[2])
    print("for exercise three:")

    for M in[4, 8, 16, 36]:
        U3 = fivePoints(0, math.pi, 0, 0.5 * math.pi, M, M, ex=3)
        print()
        print("x轴步长：", U3[0])
        print("y轴步长：", U3[1])
        print("误差：", U3[2])
    '''






