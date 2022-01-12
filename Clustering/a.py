import operator
import numpy as np
import pandas as pd


def readFile(filename):
    data = []
    try:
        file = open(filename, 'r')
    except Exception:
        print("Error: 没有找到文件或读取文件失败")
        return None
        pass
    else:
        file_data = file.readlines()
        i=0
        for row in file_data:

            if(i==0):
                i = i + 1
                continue

            # if " " in row:
            #     tmp_list = row.split(' ')
            # elif "," in row:
            #     tmp_list = row.split(',')
            # else:
            #     tmp_list = row.split('\t')  # 按空格切分成每行数据
            # tmp_list = row.split(' ')  # 按空格切分成每行数据
            tmp_list = row.split(',')
            tmp_list[-1] = tmp_list[-1].replace('\n', '')  # 去掉换行符
            data.append(tmp_list)
        return data
def equivalence_class_division(data):
    a_all = []
    for row in data:
        a = []
        for i, element in enumerate(data):
            if (operator.eq(element, row)):
                a.append(i)
        if a not in a_all:
            a_all.append(a)
    return a_all
#数据处理
if __name__ == "__main__":
    filename = "CASP(Physicochemical Properties of Protein Tertiary Structure)111.csv"
    # start = time.perf_counter()
    data = readFile(filename)#data里面已经不包括第一行
    print(data[0])

    for line in range(len(data[0])-1):
        print(line)
        data_line = [example[line] for example in data]  # 第i列
        print(data_line)
        equ_class = equivalence_class_division(data_line)
        print(equ_class)
        for j in equ_class:
            #空缺值不发生改变
            if data[j[0]][line] == '?':
                print("111")
                continue
            for i in j:
                data[i][line]=equ_class.index(j)
    # print(data)

    array = np.array(data)
    save = pd.DataFrame(array)
    save.to_csv('../data_new/' + filename, index=False, header=False, sep="\t")

