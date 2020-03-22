import string

import pandas as pd

"""
Series是带标签的数组，一个index和一个value，用Series.index,Series.value获取和操作
Series里的缺失值是NaN（not a number），NaN默认是float类型
"""

if __name__ == '__main__':
    s = pd.Series(range(5), index=list("abcde"))
    print("创建的Series:\n", s)
    print(s["a"])  # 用标签检索
    print(s[["a", "b"]])  # 用多个标签检索
    print(s[:2])  # 用index检索
    print(s[s > 3])  # 用boolean值检索

    dic = {"name": "lixiang", "age": 32, "country": "china"}
    info = pd.Series(dic)
    print(info)
    print(info["age"])

    # 这里的gender在dic里没有，创建的info1里就是NaN
    info1 = pd.Series(dic, index=["name", "age", "gender"])
    print(info1)

    # a里的value的dtype是int64
    dic1 = {string.ascii_lowercase[i]: i for i in range(10)}
    print(dic1)
    a = pd.Series(dic1)
    print(a)
    # b里的value的dtype就成了float64，
    # 因为index和dic1里的key没有完全匹配上，匹配不上的就是NaN，NaN的类型是float
    b = pd.Series(dic1, index=list(string.ascii_lowercase[5:15]))
    print(b)

    # 更改类型
    a.astype(float)
