import numpy as np
import math


def true(x, t, pro=1):

    if pro == 1:
        return np.cos(math.pi*t) * np.sin(math.pi*x)

    elif pro == 2:
        return np.sin(t) * np.sin(4*math.pi*x)


def ex3(h, tao, pro=1):
    xmin = 0
    xmax = 1
    tmin = 0
    tmax = 1
    xs = np.arange(xmin, xmax+h, h)
    ts = np.arange(tmin, tmax+tao, tao)
    if pro == 1:
        a = 1
        def u0(_i):
            _x =xs[_i]
            return np.sin(math.pi*_x)
        def ut0(_i):
            return 0
    elif pro == 2:
        a = 1 / (16*math.pi**2)
        def u0(_i):
            return 0
        def ut0(_i):
            _x = xs[_i]
            return np.sin(4*math.pi*_x)

    r = a*tao / h
    print("r={}" .format(r), end=',')

    U = np.zeros((len(ts), len(xs)))
    print("网格节点数（时间层，空间层）:",U.shape)

    for x in xs:
        i = list(xs).index(x)
        U[0][i] = u0(i)
    for x in xs:
        i = list(xs).index(x)
        U[1][i] = 0.5 * r**2 * (u0(i-1) + u0(i+1)) + (1 - r**2) * u0(i) + tao * ut0(i)
    for k in ts[1:-1]:
        ki = list(ts).index(k)
        for j in xs[1:-1]:
            ji = list(xs).index(j)
            U[ki+1][ji] = r**2 * (U[ki][ji-1] + U[ki][ji+1]) + 2*(1-r**2)*U[ki][ji] - U[ki-1][ji]

    #print(U)

    UTrue = np.zeros((len(ts), len(xs)))
    for x in xs:
        for t in ts:
            UTrue[list(ts).index(t)][list(xs).index(x)] = true(x, t, pro=pro)
    #print(UTrue)

    print("误差：", np.linalg.norm(U - UTrue))

'''
print("r > 1, 不稳定：")
ex3(0.1, 0.2, pro=1)
ex3(0.05, 0.1, pro=1)
ex3(0.005, 0.01, pro=1)

print("r = 1")
ex3(0.2, 0.2, pro=1)
ex3(0.1, 0.1, pro=1)
ex3(0.05, 0.05, pro=1)
ex3(0.01, 0.01, pro=1)

print("r < 1, 稳定：")
ex3(0.2, 0.1, pro=1)
ex3(0.1, 0.05, pro=1)
ex3(0.01, 0.005, pro=1)
'''
a2 = 16*math.pi**2
print("r > 1, 不稳定：")
ex3(0.0012, 0.2, pro=2)
ex3(0.00063, 0.1, pro=2)
#ex3(0.005, 0.01, pro=2)

print("r < 1, 稳定：")
ex3(0.5, 0.5, pro=2)
ex3(0.25, 0.25, pro=2)
#ex3(2/a2, 1, pro=2)
