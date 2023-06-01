# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : factor_ops.py
# Time       : 2022/9/12 17:55
# Author     : Daic
"""
import numpy as np
import pandas as pd
import talib as ta
from scipy.stats import norm
from factors.ops.rolling import _wma, _max_distance, _min_distance, _alpha191_143

# from calculator.base import FactorBase
from typing import List, Union, Dict, Any

try:
    from qlib.data.ops import rolling_slope, rolling_rsquare, rolling_resi
    from qlib.data.ops import expanding_slope, expanding_rsquare, expanding_resi
except ImportError:
    print("部分因子依赖cython实现的滚动回归, 导入 qlib.data.ops.rolling_* 失败！")
    print("如果不想安装qlib, 请使用将 https://github.com/microsoft/qlib/tree/main/qlib/data/_libs 下的pyx文件编译")


def RANK(data: pd.DataFrame):
    return data.rank(axis=1, pct=True)


def IFELSE(condition: pd.DataFrame, A: Union[pd.DataFrame, Any], B: Union[pd.DataFrame, Any]):
    """
    条件选择
    """
    if isinstance(A, pd.DataFrame):
        na_map = A.isna().astype(float).replace(1., np.nan)
        return A.where(condition, B) + na_map
    elif isinstance(B, pd.DataFrame):
        na_map = B.isna().astype(float).replace(1., np.nan)
        return B.where(~condition, A) + na_map
    else:
        raise ValueError("A, B 必须有一个是DataFrame")


def COUNT(condition: pd.DataFrame, n: int, na_map: pd.DataFrame = None):
    """
    滚动计数
    """
    condition = condition.astype(int)
    if na_map is not None:
        condition = condition + na_map
    return condition.rolling(n).sum()


def SMA(data: pd.DataFrame, n: int, m: int, ignore_nan=True):
    assert n > m
    na_map = data.isna().astype(float).replace(1., np.nan)
    return data.ewm(alpha=m / n, ignore_na=ignore_nan).mean() + na_map


def WMA(data: pd.DataFrame, n: int):
    weight = np.arange(n) + 1
    weight = weight / weight.sum()
    idx = data.index
    col = data.columns
    return pd.DataFrame(_wma(data.values, weight), index=idx, columns=col)


def DECAYLINEAR(data: pd.DataFrame, n: int):
    """
    线性衰减移动平均加权
    """
    weight = np.array([2 * i / (n * (n + 1)) for i in range(1, n + 1)])
    idx = data.index
    col = data.columns
    return pd.DataFrame(_wma(data.values, weight), index=idx, columns=col)


def DELTA(A: pd.DataFrame, n: int):
    return A.diff(n)


def DELAY(A: pd.DataFrame, n: int):
    return A.shift(n)


def SUM(A: pd.DataFrame, n: int):
    return A.rolling(n).sum()


def MEAN(price: pd.DataFrame, N: int):
    """
    滚动平均
    """
    return price.rolling(N).mean()


def CORR(A: pd.DataFrame, B: pd.DataFrame, n: int):
    return A.rolling(n).corr(B)


def COVARIANCE(A: pd.DataFrame, B: pd.DataFrame, n: int, sign=False):
    """
    滚动协方差
    """
    if sign:
        return np.sign(A.rolling(n).cov(B))
    else:
        return A.rolling(n).cov(B)


def MAX(A: pd.DataFrame, B):
    """
    两个序列的最大值
    """
    return np.maximum(A, B)


def MIN(A: pd.DataFrame, B: pd.DataFrame):
    """
    两个序列的最大值
    """
    return np.minimum(A, B)


def HIGHDAY(A: pd.DataFrame, N: int, zero_diff=False):
    """
    N日内最高价的距离
    """
    idx = A.index
    col = A.columns
    na_map = A.isna().astype(float).replace(1., np.nan)
    A = A.fillna(-np.inf)
    return pd.DataFrame(_max_distance(A.values, N), index=idx, columns=col) - int(zero_diff) + na_map


def LOWDAY(A: pd.DataFrame, N: int, zero_diff=False):
    """
    N日内最低价的距离
    """
    idx = A.index
    col = A.columns
    na_map = A.isna().astype(float).replace(1., np.nan)
    A = A.fillna(+np.inf)
    return pd.DataFrame(_min_distance(A.values, N), index=idx, columns=col) - int(zero_diff) + na_map


def TS_MAX(A: pd.DataFrame, N: int):
    """
    N日内最大值
    """
    return A.rolling(N).max()


def TS_MIN(A: pd.DataFrame, N: int):
    """
    N日内最小值
    """
    return A.rolling(N).min()


def TS_RANK(data: pd.DataFrame, N: int):
    """
    滚动排名
    """
    return data.rolling(N).rank(pct=True)


def SHARPE(price: pd.DataFrame, N: int):
    """
    滚动夏普比率
    """
    return price.rolling(N).mean() / (price.rolling(N).std() + 1e-7)


def OMEGA_RATIO(price: pd.DataFrame, N: int, required_return_daily=0.0):
    """
    滚动OMEGA比率
    """
    ret = (price / price.shift() - 1) - required_return_daily
    pos = (ret * (ret > 0)).rolling(N).sum()
    neg = - (ret * (ret < 0)).rolling(N).sum()
    return pos / (neg + 1e-7)


def VaR(price: pd.DataFrame, N: int, p: float = 0.05, method: str = "param"):
    """
    滚动VaR
    @NOTE:数据N越长, 分布越正态则越有意义
    https://juejin.cn/post/7106772397033783310
    """
    assert method in ["param", "historical"]
    ret = price / price.shift() - 1
    na_map = ret.isna().astype(float).replace(1., np.nan)

    if method == "historical":
        return ret.rolling(N).quantile(p) + na_map
    else:
        return ret.rolling(N).mean() + ret.rolling(N).std() * norm.ppf(p) + na_map


def CVaR(price: pd.DataFrame, N: int, p: float = 0.05, method: str = "param"):
    ret = price / price.shift() - 1
    var = VaR(price, N, p, method)
    cvar = (ret * (ret <= var)).rolling(N).mean()
    return cvar


def HL_STD(high: pd.DataFrame, low: pd.DataFrame, N: int):
    """
    每日波幅的滚动标准差
    """
    return (high - low).rolling(N).std()


def STD(price: pd.DataFrame, N: int):
    """
    滚动标准差
    """
    return price.rolling(N).std()


def MDD(close: pd.DataFrame, high: pd.DataFrame, N: int, keep_positive: bool = True):
    """
    N日内价格最大回撤
    """
    mdd = close / high.rolling(N).max() - 1
    if not keep_positive:
        mdd = mdd.where(mdd > 0, 0)
    return mdd + 1e-7


def REGBETA(X: pd.DataFrame, Y: pd.DataFrame, N: int):
    """
    线性回归斜率
    """
    return X.rolling(N).cov(Y) / X.rolling(N).var()


def REGRESI(X: pd.DataFrame, Y: pd.DataFrame, N: int):
    """
    线性回归截距
    """
    return X.rolling(N).mean() - Y.rolling(N).mean() * REGBETA(X, Y, N)


# =============================== ta-lib ops =============================
def LINEARREG(X: pd.DataFrame, N: int):
    """
    线性回归值
    """
    return X.apply(lambda x: ta.LINEARREG(x, N))


def LINEARREG_SLOPE(X: pd.DataFrame, N: int):
    """
    线性回归斜率
    """
    return X.apply(lambda x: ta.LINEARREG_SLOPE(x, N))


def LINEARREG_ANGLE(X: pd.DataFrame, N: int):
    """
    线性回归角度
    """
    return X.apply(lambda x: ta.LINEARREG_ANGLE(x, N))


def LINEARREG_INTERCEPT(X: pd.DataFrame, N: int):
    """
    线性回归截距
    """
    return X.apply(lambda x: ta.LINEARREG_INTERCEPT(x, N))


def TSF(X: pd.DataFrame, N: int):
    """
    时间序列预测(最小二乘法回归)
    """
    return X.apply(lambda x: ta.TSF(x, N))
