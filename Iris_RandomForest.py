#!/user/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = "data/iris.data"
    data = pd.read_csv(path, header=None)
    x = data[list(range(4))]
    y = pd.Categorical(data[4]).codes
    print("特征数据：\n", x.head, "y = \n", y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    plt.figure(figsize=(6, 4), facecolor='#FFFFFF')
    for i, pair in enumerate(feature_pairs):
        x_train = X_train[pair]
        x_test = X_test[pair]

        # 训练决策树模型
        model = RandomForestClassifier(100, criterion="entropy", max_depth=3, oob_score=True)
        # oob_score 是否对袋外数据进行打分 out of bag
        model.fit(x_train, y_train)

        # 画图
        X1_min, X2_min = x_train.min()
        X1_max, X2_max = x_train.max()
        t1 = np.logspace(X1_min, X1_max, 500)
        t2 = np.logspace(X2_min, X2_max, 500)

        # 生成网格点矩阵，https://blog.csdn.net/lllxxq141592654/article/details/81532855
        x1, x2 = np.meshgrid(t1, t2)  # x1(500,500) x2(500,500)
        x_show = np.stack((x1, x2), axis=1)  # 沿着新轴连接数组的序列 https://blog.csdn.net/wgx571859177/article/details/80987459
        # X_show.shape = (500,2,500)

        # 训练集上的精度
        y_train_pred = model.predict(x_train)
        acc_train = accuracy_score(y_train, y_train_pred)
        # 测试集上的精度
        y_test_pred = model.predict(x_test)
        acc_test = accuracy_score(y_test, y_test_pred)
        print("特征：({} , {}):  oob Score={:.2%}".format(pair[0], pair[1], model.oob_score_))
        print("训练集精度 = {:.2%} ,测试集精度 = {:.2%}\n".format(acc_train, acc_test))
