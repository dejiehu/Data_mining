from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
def readfile():
    # my_data = np.loadtxt("../data_new/Concrete Slump Test.csv")
    my_data = np.loadtxt("../data_new/servo.csv")
    # print(my_data)
    return my_data

def deal_data(my_data, m, n):  # 处理数据表  找出条件属性和决策属性用
    if n + 1 > m:
        for d in range(n, m - 1, -1):
            my_data = np.delete(my_data, d, 1)  # d为下标
    return my_data


if __name__ == '__main__':
    my_data = readfile()
    dec_data = deal_data(my_data, 0, my_data.shape[1] - 2)
    K = 4
    # x = np.array([10.1,10.9,10.4,10.4,9.6,9.8,9.7,0.1,5.2,1.2,6.5])
    # y = x.reshape(-1,1)
    y_pred = KMeans(n_clusters = K,max_iter= 600).fit_predict(dec_data)

    # center = y_pred.cluster_centers_
    # print(center)
    print(y_pred,"划分结果")
    class_list = [[]] * K
    for i in range(len(y_pred)):
        class_list[y_pred[i]] = class_list[y_pred[i]] + [dec_data[i][0]]
        # class_list[y_pred[i]] = class_list[y_pred[i]] + [x[i]]
    print(class_list)
    # print(x)
    # print(y)
    # plt.scatter(y[:, 0], y[:, 1], marker='o')
    # plt.show()