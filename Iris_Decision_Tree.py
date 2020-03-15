import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pydotplus

iris_feature_E = "sepal length", "sepal width", "petal length", "petal width"
iris_feature = "花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"
iris_class = "Iris-setosa", "Iris-versicolor", "Iris-virginica"

if __name__ == '__main__':
    data = pd.read_csv("data/iris.data")
    print("数据集大小：", data.shape)
    x = data.iloc[:, 0:4]
    # 把y转成0，1，2
    y = pd.Categorical(data.iloc[:, 4]).codes

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
    model = DecisionTreeClassifier(criterion="entropy", min_samples_split=10)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))  # 输出精度，正确率

    # 关于精度的算法
    y_test_hat = model.predict(X_test)
    print("在测试集上预测的结果：", y_test_hat, "\n测试集的正确结果：", y_test)
    result = y_test_hat == y_test
    acc = np.mean(result)
    print("在测试集上的精度为：", acc)  # 和上面model.score的结果一样

    # 用交叉验证来选择决策树的一些超参数
    max_depth_can = list(range(1, 10))
    params = {"criterion": ["entropy", "gini"],
              "max_depth": max_depth_can,
              "min_samples_split": list(range(5, 50, 5))}
    print(params)
    model = DecisionTreeClassifier()
    decision_tree_mode = GridSearchCV(model, param_grid=params)
    decision_tree_mode.fit(X_train, y_train)  # 要先训练才能输出下一步的值
    print("决策树最优参数：", decision_tree_mode.best_params_,
          '最优分数:%.2f%%' % (100 * decision_tree_mode.best_score_))

    # 也可以自己做一下
    # depth = np.arange(1, 15)
    # err_list = []
    # for d in depth:
    #     clf = DecisionTreeClassifier(criterion="entropy", max_depth=d)
    #     clf.fit(X_train, y_train)
    #     y_test_hat = clf.predict(X_test)
    #     result = np.mean(y_test == y_test_hat)
    #     error = 1 - np.mean(result)
    #     err_list.append(error)
    #     print("树深度：", d, "错误率：", error)
    # plt.figure(facecolor='w')
    # plt.plot(depth, err_list, "ro-", lw=2)
    # plt.xlabel("决策树的深度")
    # plt.ylabel("测试集错误率")
    # plt.title("决策树的深度与过拟合", fontsize=15)
    # plt.grid(b=True, ls=":", color='#606060')
    # plt.show()

    # 画出决策树图
    # 1、输出.dot文件格式，可以用
    with open("result/iris.dot", "w") as f:
        tree.export_graphviz(model, out_file=f)
    # 2、输出文件
    tree.export_graphviz(decision_tree_mode, out_file="result/iris.dot")
    # 3、输出pdf格式
    dot_data = tree.export_graphviz(model,
                                    out_file=None,
                                    feature_names=iris_feature_E,
                                    class_names=iris_class, filled=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("result/iris.pdf")
    f = open("result/iris.png", "wb")
    f.write(graph.create_png())
    f.close()
