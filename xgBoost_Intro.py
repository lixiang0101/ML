import numpy as np
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

"""
agaricus_train.txt 数据说明：
    每一行表示一个样本，第一列是y值：1表示有毒蘑菇，0表示无毒蘑菇
    第1行第2列：2:1，表示：第1个样本的的第2纬度是1；9:1 表示：第1个样本的第9列是1，其它每标注的是0
    没有标注为1的都是0
"""


def error_rate(y_hat, y):
    return "error", float(sum((y_hat > 0.5) != y.get_label())) / len(y_hat)


if __name__ == '__main__':
    data_train = xgb.DMatrix("data/agaricus_train.txt")
    data_test = xgb.DMatrix("data/agaricus_test.txt")
    print(data_train.num_row())

    # "objective": "binary:logistic" 用logistic回归做二分类;'objective': 'reg:logistic' 做回归
    # "silent": 1 不输出更多的信息
    # "eta": 1 每颗数的权值，xgboost自己有一个传值，这里是人为再给一个权值，防止学的太快
    param = {"max_depth": 3, "eta": 1, "silent": 1, "objective": "binary:logistic"}
    watchlist = [(data_test, "eval"), (data_train, "train")]
    n_round = 7
    bst = xgb.train(params=param, dtrain=data_train, num_boost_round=n_round, evals=watchlist, feval=error_rate)

    y_test_hat = bst.predict(data_test)
    y_test = data_test.get_label()
    print(y_test_hat > 0.5, y_test)
    error = sum((y_test_hat > 0.5) != y_test)
    error_rate_test = error / len(y_test)
    print("测试集样本总数：", len(y_test_hat))
    print("预测错误样本：\t%4d" % error)
    print("测试集上的预测错误率为：\t%.5f%%" % (error_rate_test * 100))
