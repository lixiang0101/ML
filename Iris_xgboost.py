import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    data_path = "data/iris.data"
    data = pd.read_csv(data_path, header=None)
    print(data)
    x = data.iloc[:, :3]
    y = data[4]
    y = pd.Categorical(y).codes

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
    data_train = xgb.DMatrix(X_train, label=y_train)
    data_test = xgb.DMatrix(X_test, label=y_test)
    param = {"max_depth": 3, "eta": 0.4, "silent": 1,
             "objective": "multi:softmax", "num_class": 3}
    watchlist = [(data_test, "eval"), (data_train, "train")]
    bst = xgb.train(params=param, num_boost_round=6, dtrain=data_train, evals=watchlist)
    # 保存模型
    bst.save_model("result/iris_xgboost.model")

    y_test_hat = bst.predict(data_test)
    result = y_test == y_test_hat
    print("测试集正确率：\t%.2f%%" % (float(np.sum(result)) / len(y_test) * 100))

    # 与logistic和RandomForest对比
    # LogisticRegressionCV可以自己通过交叉验证选择正则化C，Cs=10，会产生10个等间距的C值
    models = [("LogisticRegressionCV", LogisticRegressionCV(Cs=10, cv=3)),
              ("RandomForestClassifier", RandomForestClassifier(n_estimators=30, criterion="gini", max_depth=3))]

    for name, model in models:
        model.fit(X_train, y_train)
        x_test_hat = model.predict(X_test)
        print("模型：", name, "auc：%.2f" % accuracy_score(y_test, x_test_hat))
