from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet
import numpy as np
import pandas as pd

if __name__ == '__main__':
    print("hello world")
    data = pd.read_csv("data/Advertising.csv")[["TV", "Radio", "Newspaper", "Sales"]]
    print(data.head())
    X = data[["TV", "Radio", "Newspaper"]]
    y = data["Sales"]
    print("X = \n", X)
    print("y = \n", y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=0.7)
    model = Lasso()
    alpha_can = np.logspace(-3, 2, 10)
    np.set_printoptions(suppress=True)
    print("alpha：", alpha_can)
    # 做交叉验证，选出超参数
    lasso_model = GridSearchCV(model, param_grid={"alpha": alpha_can})
    lasso_model.fit(x_train, y_train)
    print("lasso模型的超参数：", lasso_model.best_estimator_)

    y_hat = lasso_model.predict(x_test)
    score = lasso_model.score(x_test, y_test)  # 计算的是R方值
    print(score)
    MSE = np.average((y_hat - np.array(y_test)) ** 2)
    rMSE = np.sqrt(MSE)
    print(MSE, rMSE)
