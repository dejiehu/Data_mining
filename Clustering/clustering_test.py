

# import numpy as np
# import matplotlib.pyplot as plt
#
# from sklearn.datasets import make_blobs
# # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
# X, y = make_blobs(n_samples=10, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2],
#                   random_state =9)
#
# print(X)
# # print(y)
# plt.scatter(X[:, 0], X[:, 1], marker='o')
# plt.show()
#
# from sklearn.cluster import KMeans
# y_pred = KMeans(n_clusters=4).fit_predict(X)
# print(y_pred)
#
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from sklearn.datasets import load_iris
import random


class K_Means():

    def __init__(self,X_data,K):

        self.X_data=X_data
        self.K=K  # 簇中心数

        self.datanum=X_data.shape[0]  # 数据的数目
        print(self.datanum,"数据的数目")

        self.dim=X_data.shape[-1]  # 数据维度

        self.AllCenter=[]  # 存储迭代过程中所有的簇中心
        self.ALLCluster=[]  # 存储迭代过程中所有的聚类结果
        self.AllJ=[]       # 存储迭代过程在所有的误差函数

        self.oldCenter=np.zeros((self.K, self.dim)) # 存储前一次迭代的簇中心

        self.newCenter=self.InitCenter_jia()  # 将簇中心初始化，这个newCenter属性用来存储本次迭代的簇中心
        self.AllCenter.append(self.newCenter)

        self.iternum=0 # 记录迭代次数

        while not (self.oldCenter == self.newCenter).all(): # 如果前后两次簇中心没变化，则迭代结束
            if self.iternum > 12:  # 如果迭代次数大于十二，结束
                break
            print("itter  ",self.iternum)
            self.distance=self.calculate_distance()
            self.oldCenter=self.newCenter
            self.newCenter,self.cluster=self.update_Center()
            self.J=self.calculate_J()
            if (self.oldCenter == self.newCenter).all():
                print("迭代结束")
                break
            self.iternum += 1

            self.AllCenter.append(self.newCenter)  # 往存储簇中心的列表中添加迭代后得到的簇中心
            self.ALLCluster.append(self.cluster)
            self.AllJ.append(self.J)
            print("newCenter", self.newCenter)
            print()

    #  初始化簇中心KMeans
    def InitCenter(self):
        Center = np.zeros((self.K, self.dim))
        rand_list= random.sample(list(range(0, self.datanum)), self.K) # 从所有样本中挑出K个作为簇中心
        for i in range(self.K):
            print( self.X_data[rand_list[i]])
            Center[i] = self.X_data[rand_list[i]]
            print(Center[i])
        print(Center)
        return Center

    # 初始化簇中心KMeans++
    def InitCenter_jia(self):
        Center=np.zeros([self.K,self.dim])
        num=0  # 用于记录已知簇中心个数
        centerx=self.X_data[random.randint(0,self.datanum-1)] # 先随机初始化一个簇中心
        Center[num]=centerx
        num+=1

        while num<self.K:
            dis = np.zeros([self.datanum,num])
            for i in range(0,self.datanum): # 计算所有样本到所有已知簇中心的距离
               for j in range(0,num):
                   dis[i][j]=np.linalg.norm(self.X_data[i]-Center[j])
            min_dis_For_Data=np.min(dis,axis=1) # 取每个样本到与其最近的簇中心的距离
            All_dis=np.sum(min_dis_For_Data)  # 计算总距离
            ranDis=All_dis*random.random()    # 产生一个0~All_dis之间的数

            for i in range(0,self.datanum):
                ranDis-=min_dis_For_Data[i]
                if ranDis<=0:
                    Center[num]=self.X_data[i]
                    num+=1
                    break
        return Center

    # 数据到簇中心距离的矩阵
    def calculate_distance(self):
        distance=np.zeros([self.datanum, self.K])
        for i in range(0,self.datanum):
            for k in range(0,self.K):
                distance[i, k] = np.linalg.norm(self.X_data[i, :]-self.newCenter[k])
        # print(self.X_data[0, :] , self.newCenter[k],self.X_data[0, :] - self.newCenter[k],"self.X_data[0, :]")
        # print(np.linalg.norm(self.X_data[0, :] - self.newCenter[k]))
        return distance

    # 更新簇中心
    def update_Center(self):
        center = np.zeros([self.K,self.dim])
        cluster = np.argmin(self.distance, axis=1)  # 数据点所属簇的矩阵
        KongCu_List = [] # 存储空簇序号的列表
        for i in range(0,self.K):
            data = self.X_data[cluster==i]  # 属于第i个聚类中心的数据
            print("第%d个簇的点数 "%i,data.shape[0])
            if data.shape[0] != 0:  # 如果不是空簇
                center_i=np.mean(data,axis=0) # 寻找属于同一个簇中的样本点的中心作为新的簇中心
                center[i]=center_i
            else:
                print("第%d个簇是空簇！！！"%i)
                KongCu_List.append(i)

        # 寻找离当前已知簇中心最远的点作为空簇簇中心
        while len(KongCu_List) != 0: # 循环直至不存在空簇
            Centernum = list(range(0,self.K))
            NotnullCenter_index =[i for i in Centernum if i not in KongCu_List]  # 挑出非空簇的索引
            NotnullCenter = []  # 存储非空簇的簇中心

            for i in NotnullCenter_index:
                NotnullCenter.append(center[i])
            dis = np.zeros([self.datanum,len(NotnullCenter)])
            # 计算所有样本到非空簇的距离
            for i in range(0, self.datanum):
                for k in range(0, len(NotnullCenter)):
                    dis[i][k] = np.linalg.norm(self.X_data[i]-NotnullCenter[k])

            maxdis=np.sum(dis, axis=1)  # 求距离和
            maxdisarg=np.argmax(maxdis)  # 找最大距离的样本的索引
            center[KongCu_List.pop()]=self.X_data[maxdisarg]  # 将其对应的样本作为空簇中心

        return center, cluster

    # 计算SSE
    def calculate_J(self):
        J=0
        for i in range(0,self.datanum):
            J+=np.linalg.norm((self.X_data[i]-self.oldCenter[self.cluster[i]]))**2
        # print(self.X_data[0]-self.oldCenter[self.cluster[0]],self.X_data[0],self.oldCenter[self.cluster[0]],"self.X_data[i]-self.oldCenter[self.cluster[i]]")
        # print(np.linalg.norm(self.X_data[0]-self.oldCenter[self.cluster[0]])**2)
        return J

    def Visual(self):
        mark = ['or', 'ob', 'og', 'om', 'oy', 'oc']  # 簇中点的颜色及形状
        center =['Dr', 'Db', 'Dg', 'Dm', 'Dy', 'Dc'] # 簇中心颜色及形状
        figure=plt.figure(figsize=(8,7))

        if self.dim == 1:  # 如果数据维度是1
            ax1 = figure.add_subplot(221)
            ax1.scatter([0]*self.datanum,self.X_data.tolist(), s=3)
            plt.title("未聚类前散点图")

            plt.ion()
            ax3 = figure.add_subplot(212)
            plt.title("误差函数图")

            ax2 = figure.add_subplot(222)
            x = list(range(self.iternum))
            for t in range(0, self.iternum):
                ax2.cla()
                j = 0
                ax3.plot(x[t], self.AllJ[t], "b.")  # 打印每次迭代的SSE
                for i in range(0,self.K):  # 打印簇中心
                    ax2.plot([0], self.AllCenter[t][i].tolist(), center[i], markersize=5, zorder=2)

                if t == self.iternum - 1:
                    plt.title("最终聚类结果散点图")
                else:
                    plt.title("第%d次迭代的聚类结果" % t)

                for i in self.ALLCluster[t]:
                    ax2.plot(self.X_data[j:j + 1, 0].tolist(), mark[i], markersize=3, zorder=1)
                    j += 1
                plt.pause(1.5)  # 停1.5s再打印下一次迭代结果
            ax3.plot(self.AllJ, "g-")
            plt.ioff()
            plt.show()

        if self.dim == 2:
            ax1 = figure.add_subplot(221)
            ax1.scatter(self.X_data[:,0].tolist(), self.X_data[:,1].tolist(),s=3)
            plt.title("未聚类前散点图")

            plt.ion()
            ax3 = figure.add_subplot(212)
            plt.title("误差函数图")

            ax2 = figure.add_subplot(222)
            x=list(range(self.iternum))
            for t in range(0,self.iternum):
                ax2.cla()
                j = 0
                ax3.plot(x[t], self.AllJ[t],"b.")
                for i in range(0,self.K):

                    ax2.plot(self.AllCenter[t][i,0],self.AllCenter[t][i,1],center[i],markersize=5,zorder=2)  # zorder越大，越在上层显示

                if t==self.iternum-1:
                    plt.title("最终聚类结果散点图")
                else:
                    plt.title("第%d次迭代的聚类结果"  % t)

                for i in self.ALLCluster[t]:

                    ax2.plot(self.X_data[j:j+1,0].tolist(),self.X_data[j:j+1,1].tolist(),mark[i],markersize=3,zorder=1)
                    j+=1
                plt.pause(1.5)
            ax3.plot(self.AllJ, "g-")
            plt.ioff()
            plt.show()

        if self.dim==3:
            ax1 = figure.add_subplot(221,projection='3d')
            ax1.scatter(self.X_data[:, 0].tolist(), self.X_data[:, 1].tolist(),self.X_data[:,2].tolist(),s=3)
            plt.title("未聚类前散点图")

            plt.ion()
            ax3 = figure.add_subplot(212)
            plt.title("误差函数图")

            ax2 = figure.add_subplot(222,projection='3d')
            x = list(range(self.iternum))

            for t in range(0, self.iternum):
                ax2.cla()
                j = 0
                ax3.plot(x[t], self.AllJ[t], "b.")

                ax2.plot(self.AllCenter[t][:, 0].tolist(), self.AllCenter[t][:, 1].tolist(),
                         self.AllCenter[t][:, 2].tolist(), "k*", label='聚类中心', markersize=5, zorder=2)
                plt.legend()
                if t == self.iternum - 1:
                    plt.title("最终聚类结果散点图")
                else:
                    plt.title("第%d次迭代的聚类结果" % t)

                for i in self.ALLCluster[t]:
                    ax2.plot(self.X_data[j:j + 1, 0].tolist(), self.X_data[j:j + 1, 1].tolist(),self.X_data[j:j + 1, 2].tolist(), mark[i], markersize=3, zorder=1)
                    j += 1
                plt.pause(1.5)
            ax3.plot(self.AllJ, "g-")
            plt.ioff()
            plt.show()


def example0():
    N=1000
    C=[[N/4,N/2,0,N/2],[N/2,N,0,N/2],[N/4,N/2,N/2,N],[N/2,N,N/2,N]]
    data=[]
    for i in range(4):
        center_datanum=random.randint(20,50)
        for j in range(center_datanum):
            change=random.randint(20,100)
            x=random.randint(C[i][0]+change,C[i][1]-change)
            y=random.randint(C[i][2]+change,C[i][3]-change)
            data.append([x,y])
    data=np.mat(data)
    print(data)
    test=K_Means(data,4)
    # test.Visual()


def example1():
    x1 = np.zeros((10, 1))
    x2 = np.zeros((10, 1))
    for i in range(0, 10):
        x1[i] = np.random.rand() * 4
        x2[i] = np.random.rand() * 5 + 5
        x = np.append(x1, x2, axis=0)
    print(x)
    # test = K_Means(x, 2)
    # test.Visual()


def example2():
    data=load_iris().data[: , 2:]  # 取iris数据集后两列
    test=K_Means(data,3)
    test.Visual()


def example3():
    data=load_iris().data[:,1:]  # 取iris数据集后三列
    test=K_Means(data,3)
    test.Visual()



if __name__ == '__main__':
    example0()
    # example1()
    # example2()
    # example3()
