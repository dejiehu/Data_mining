import math
from math import sqrt
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def Gen_dataSet():
    X,y = make_moons(n_samples=200, noise=0.09)
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.show()
    return X

#欧氏距离
def distnace(x1,x2):
    return sqrt(math.pow(x1[0] - x2[0],2) + math.pow(x1[1] - x2[1],2))

#每个点都要计算k-距离，在计算一个点的k-距离的时候，首先要计算该点到其他所有点的欧几里德距离，按照距离升序排序后，选择第k小的距离作为k-距离的值
def k_distance_group(dataSet,k):
    x = []
    dist_list = []
    dist_Matrix =[[0 for j in range(len(dataSet))] for i in range(len(dataSet))]
    for i in range(len(dataSet)):
        for j in range(len(dataSet)):
            dist_Matrix[i][j] = (distnace(dataSet[i],dataSet[j]))
    for i in range(len(dist_Matrix)):
        dist_Matrix[i].sort()
        x.append(i)
        dist_list.append(dist_Matrix[i][k])
    dist_list.sort()
    plt.figure()
    plt.plot(x, dist_list)
    plt.show()

#聚类算法
def DBSCAN(dataSet,Eps,MinPts):
    k = -1
    neighbor_list = []  # 用来保存每个数据的邻域
    omega_list = []  # 核心对象集合
    gama = set([x for x in range(len(dataSet))])  # 初始时将所有点标记为未访问
    cluster = [-1 for _ in range(len(dataSet))]  # 聚类
    for i in range(len(dataSet)):
        neighbor_list.append(find_neighbor(i, dataSet, Eps))  # 找 < eps的点加入到neighbor_list
        if len(neighbor_list[-1]) >= MinPts:
            omega_list.append(i)  # 将样本加入核心对象集合
    omega_list = set(omega_list)  # 转化为集合便于操作
    while len(omega_list) > 0:  # 当核心对象集合不为空
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))  # 随机选取一个核心对象
        k = k + 1
        Q = list()
        Q.append(j)  # 将随机选取的核心对象加入到Q中
        gama.remove(j)  # 将这个核心对象从未访问的集合中移除
        while len(Q) > 0:  # Q不为空时
            q = Q[0]  # 取Q中第一个值
            Q.remove(q)  # 在从Q中移除它
            if len(neighbor_list[q]) >= MinPts:
                delta = neighbor_list[q] & gama
                deltalist = list(delta)
                for i in range(len(delta)):
                    Q.append(deltalist[i])
                    gama = gama - delta  # 将访问过得值删除
        Ck = gama_old - gama  # 得到所有访问过的点
        Cklist = list(Ck)
        for i in range(len(Ck)):
            cluster[Cklist[i]] = k  # 归类
        omega_list = omega_list - Ck
    return cluster


# 找 < eps的点
def find_neighbor(j, x, eps):
    N = list()
    for i in range(x.shape[0]):
        temp = np.sqrt(np.sum(np.square(x[j] - x[i])))  # 计算欧式距离
        if temp <= eps:
            N.append(i)
    return set(N)

if __name__ == '__main__':
    MinPts = 4
    dataSet = Gen_dataSet()
    # while 1:
    k_distance_group(dataSet, MinPts)
    Esp = eval(input('Eps:'))
    cluster = DBSCAN(dataSet, Esp, MinPts)
    plt.figure()
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=cluster)
    plt.show()