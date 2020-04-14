import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score

"""
根据葡萄酒的特征来判断葡萄酒的类别
"""

if __name__ == '__main__':
    data = pd.read_csv("data/wine.data", header=None)
    x = data.iloc[:, 1:]
    y = data[0]
    x = MinMaxScaler().fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
    print("数据集大小:\t", data.shape,
          "\n训练集大小：\t", X_train.shape,
          "\n测试集大小：\t", X_test.shape)

    # 如果logistics警告不能收敛，可以考虑对训练集做标准化
    lr = LogisticRegressionCV(cv=3, penalty="l2")
    lr.fit(X_train, y_train)
    train_score = lr.score(X_train, y_train)
    test_score = lr.score(X_test, y_test)
    print("logistics：\n",
          "训练集精度：\t", train_score,
          "测试集集度：\t", test_score)

    rf = RandomForestClassifier(n_estimators=30, criterion="gini", max_depth=3)
    rf.fit(X_train, y_train)
    y_test_hat = rf.predict(X_test)
    test_auc = accuracy_score(y_test, y_test_hat)
    print("随机森林：\n",
          "测试集accuracy_score：\t%.2f" % test_auc)

    # 特别注意，xgboost做分类的时候标签值必须从0开始
    y_train[y_train == 3] = 0
    y_test[y_test == 3] = 0
    dTrain = xgb.DMatrix(X_train, label=y_train)
    dTest = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(dTest, "eval"), (dTrain, "train")]
    param = {"max_depth": 1, "eta": 0.9, "silent": 1, "objective": "multi:softmax", "num_class": 3}
    bst = xgb.train(params=param, dtrain=dTrain, num_boost_round=6, evals=watchlist)
    y_test_hat = bst.predict(dTest)
    test_auc = accuracy_score(y_test, y_test_hat)
    print("xgboost：\n",
          "测试集accuracy_score：\t%.2f" % test_auc)

