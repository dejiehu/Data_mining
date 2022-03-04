from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def readfile(filename):
    my_data = np.loadtxt("../Adata/clustering/" + filename)
    return my_data

def deal_data(my_data, m, n):  # 处理数据表  找出条件属性和决策属性用
    if n + 1 > m:
        for d in range(n, m - 1, -1):
            my_data = np.delete(my_data, d, 1)  # d为下标
    return my_data

def res(class_list,dec_data,y_pred):
    c = class_list
    a = dec_data
    b = y_pred
    for i,n in enumerate(b):
        c[n]=c[n]+[a[i][0]]
    return c
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


if __name__ == '__main__':
    filename = 'Concrete Slump Test.csv'
    my_data = readfile(filename)
    #print('my_data',my_data)
    dec_data = deal_data(my_data, 0, my_data.shape[1] - 2)
    #print('dec_data',dec_data,type(dec_data))
    K = 8
    y_pred = KMeans(n_clusters = K,max_iter= 300).fit_predict(dec_data)
    #print(y_pred,"划分结果",set(y_pred))
    class_list = [[]] * K
    r = res(class_list, dec_data, y_pred)
    print(K,'聚类结果：',r)

    y = []
    for i in range(len(dec_data)):
        y.append(0)

    m = {0: 'o',1: 'o', 2: 'o', 3: 'o', 4: 'o',5: 'o',6: 'o',7: 'o'}
    cm = list(map(lambda dec_data: m[dec_data], y_pred))  # 将相应的标签改为对应的marker
    #print(cm)
    fig, ax = plt.subplots()
    scatter = mscatter(dec_data, y, c=y_pred, m=cm, ax=ax,cmap=plt.cm.RdYlBu)
    plt.show()