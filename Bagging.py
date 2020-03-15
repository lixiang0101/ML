import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor


def f(x):
    return 0.5 * np.exp(-(x + 3) ** 2) + np.exp(-x ** 2) + np.exp(-(x - 3) ** 2)


if __name__ == '__main__':
    # 生成模拟数据
    np.random.seed(0)
    N = 200
    x = np.random.rand(N) * 10 - 5  # 生成[-5,5]之间的随机数
    # print(x)
    x = np.sort(x)
    y = f(x) + np.random.rand(N) * 0.05
    x.shape = -1, 1 # x变成1列n行
    # print(y)

    # 经过PolynomialFeatures后会自带1，所以这里fit_intercept=False
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), fit_intercept=False)
    # PolynomialFeatures(2)：a,b两个特征，2次多项式为 [1,a,b,a^2,ab,b^2]
    # 参数interaction_only=True，不会有a^2和b^2
    # include_bias=False，没有1那项
    ridged = Pipeline([("poly", PolynomialFeatures(degree=6)), ("ridge", ridge)])
    bagging_ridged = BaggingRegressor(ridged, n_estimators=50, max_samples=0.5)
    dtr = DecisionTreeRegressor(max_depth=9)

    # 比较几个模型
    regs = [("DecisionTree", dtr),
            ("Ridged:6 degree", ridged),
            ("BaggingRidged:6 degree", bagging_ridged),
            ("BaggingDecisionTree", BaggingRegressor(dtr, 50, 0.5))]

    x_test = np.linspace(1.1 * x.min(), 1.1 * x.max(), 1000)
    # x_test: [-5.44834976 -4.23327567 -3.01820158 ...]

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.figure(figsize=(8, 6), facecolor='w')
    plt.plot(x, y, 'ro', label="训练数据")
    plt.plot(x_test, f(x_test), c='k', lw=4, ls="-", label="真实数据集")
    clrs = ("#FF2020", "m", "y", "g")
    for i, (name, reg) in enumerate(regs):
        reg.fit(x, y)
        y_test = reg.predict(x_test.reshape(-1, 1))  # reshape(-1, 1) 变成n行1列
        label = "%s,$R^2$=%.3f" % (i, reg.score(x, y))
        plt.plot(x_test, y_test, color=clrs[i], lw=i + 1, ls="-", label=label, zorder=6 - i)
    plt.legend(loc="upper left")
    plt.xlabel("X", fontsize=15)
    plt.ylabel("Y", fontsize=15)
    plt.ylim(-0.2, y.max() * 1.05)
    plt.grid(True)
    plt.show()
