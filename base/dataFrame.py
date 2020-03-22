import pandas as pd
import numpy as np

"""
DataFrame是Series容器
1、既有行索引（0轴）：index，既有列索引（1轴）：columns
2、属性
    基础属性：
        1）shape
        2）dtype  列数据类型
        3）ndim 数据纬度
        4）index 行索引
        5）column 列索引
        6）values 对象值，二维ndarrays数组
    整体属性：
        1）head(n)
        2）tail(n)
        3) info() 相关信息概览
        4）describe 基础统计
3、取行取列
    1）df[]:方括号里写数组是对行进行操作，写字符串是对列进行操作
    2) df.loc[] 根据"标签"索引行数据:
        a) df.loc[["r1","r2"],:] 或者 df.loc[:,["c1","c2"]]，冒号如果省略不写就是取所以
        b) df.loc[["r1","r1"],["c1","c2"]]
        c) df.loc["r1":"r4",["c1","c2"]] 注意：r4行也会选中
    3) df.iloc[] 根据 "位置" 获取行数据
        把df.loc[]方法里的"标签"换成数字
"""

if __name__ == '__main__':
    df = pd.DataFrame(np.arange(12).reshape(3, 4), index=list("abc"), columns=list("WXYZ"))
    print(df)
    print(df.loc["a":"b", ["W", "X"]])

    print(df.iloc[1:2, 2:3])
