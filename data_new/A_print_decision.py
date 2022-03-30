from itertools import chain


def readfileBylist(filename):
    file = open(filename,"r")
    list_row = file.readlines()
    list_data = []
    for i in range(len(list_row)):
        list_line = list_row[i].strip().split('\t')
        s=[]
        for j in range(len(list_line) ):
            s.append(int(list_line[j]))
        list_data.append(s)
    return list_data


def deal_data(my_data,m):#处理数据表   删除某一列
    del_data = [my_data[i][:] for i in range(len(my_data))]
    for d in range(len(del_data)):
        del del_data[d][m]
    return del_data

def Max_min(con_data,U_list):  #找出属性最大最小值
    Mm_list = []
    for i in range(len(con_data[0])):
        min = 10000
        Max = 0
        for j in U_list:
            if con_data[j][i] > Max:
                Max = con_data[j][i]
            if con_data[j][i] < min:
                min = con_data[j][i]
        Mm_list.append([Max,min])
    return Mm_list

def div(my_data):    #等价类的划分
    U_linkList =  [i for i in range(len(my_data))]
    Mm_list = Max_min(my_data,U_linkList)
    for i in range(len(Mm_list)):
        queue_linkList = [[]]*(Mm_list[i][0] - Mm_list[i][1] + 1)
        for j in U_linkList:
            # print(my_data[j][i] , Mm_list)
            queue_linkList[my_data[j][i] - Mm_list[i][1]] = queue_linkList[my_data[j][i] - Mm_list[i][1]] + [j]
        U_linkList.clear()
        U_linkList = list(chain.from_iterable(queue_linkList))
    div_list = []
    temp_list = [U_linkList[0]]
    for i in range(1,len(U_linkList)):
        if((my_data[U_linkList[i]] == my_data[U_linkList[i-1]])):
            temp_list.append(U_linkList[i])
            continue
        div_list.append(temp_list)
        temp_list = [U_linkList[i]]
    div_list.append(temp_list)
    return div_list


if __name__ == '__main__':
    filename = "Yeast.csv"
    list_data = readfileBylist(filename)  # 连续处理
    print(len(list_data), "对象数")
    con_data = list(map(lambda x: x[:(len(list_data[0]) - 1)], list_data))
    print(len(con_data[0]), "条件属性数")
    dec_data = list(map(lambda x: x[(len(list_data[0]) - 1):], list_data))
    dec_divlist = div(dec_data)
    print(filename)
    print("类个数：",len(dec_divlist))
    for i in dec_divlist:
        print("决策值：",dec_data[i[0]],"个数：",len(i))
