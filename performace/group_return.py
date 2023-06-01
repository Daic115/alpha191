# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : group_return.py
# Time       : 2022/9/19 1:29
# Author     : Daic
"""
import numpy as np
import numba as nb
import pandas as pd


@nb.jit(nopython=True, nogil=True)
def _tag_group(data: np.ndarray, quantile: np.ndarray, n: int):
    """
    :param data: np.ndarray (N x M)
    :param quantile: np.ndarray (N x (n-1))
    """
    N, L = data.shape
    result = np.zeros((N, L), dtype=np.float64)
    for icol in range(L):
        for iq in range(0, n - 1):
            result[:, icol] += (data[:, icol] > quantile[:, iq]).astype(np.float64)
    return result


@nb.jit(nopython=True, nogil=True)
def _quantile(X: np.ndarray, n: int):
    # 938 ms
    N, L = X.shape
    result = np.zeros((N, n - 1), dtype=np.float64)
    for i in range(N):
        x = X[i, :]
        if np.all(np.isnan(x)):
            result[i, :] = np.nan
            continue
        for i_q in range(1, n):
            result[i, i_q - 1] = np.nanquantile(x, i_q / n)
    return result


@nb.jit(nopython=True, nogil=True)
def _group_return(ret: np.ndarray, group: np.ndarray, n: int):
    """ 600ms
    :param ret: np.ndarray (N x M)
    :param group: np.ndarray (N x M)
    :param n: int
    :return: np.ndarray (N x n)
    """
    N, L = ret.shape
    result = np.zeros((N, n), dtype=np.float64)
    for i_group in range(n):
        for i in range(N):
            x = ret[i, :]
            g = group[i, :]
            result[i, i_group] = np.nanmean(x[g == i_group])
            # result[j, i] = np.nanmean(ret[j, group[j, :] == i])
    return result


def tag_group_nb(data: pd.DataFrame, n: int):
    """ 2000ms
    :param data: pd.DataFrame (N x M)
    :param N: int
    :return: pd.DataFrame (N x M)
    """
    nan_map = data.isna().astype(float).replace(1., np.nan)
    qs = _quantile(data.values, n)
    result = _tag_group(data.values, qs, n)
    return pd.DataFrame(result, index=data.index, columns=data.columns) + nan_map


def tag_group_pd(data: pd.DataFrame, n: int):
    """ 5000ms
    :param data: pd.DataFrame (N x M)
    :param N: int
    :return: pd.DataFrame (N x M)
    """
    result = data.copy()
    qs = [i / n for i in range(n + 1)]
    data_qs = [data.quantile(qs[i], axis=1) for i in range(n + 1)]
    nan_map = data.isna().astype(float).replace(1., np.nan)
    for i in range(1, n + 1):
        group = i - 1
        result[data.gt(data_qs[i - 1], axis="index") & data.le(data_qs[i], axis="index")] = group
    return result + nan_map


def _cal_group_return(ret: pd.DataFrame, group: pd.DataFrame, n: int):
    """ 1500ms
    :param ret: pd.DataFrame (N x M)
    :param group: pd.DataFrame (N x M)
    :param n: int
    :return: pd.DataFrame (N x n)
    """
    result = pd.DataFrame(np.zeros((ret.shape[0], n)), index=ret.index, columns=[f"p{i}" for i in range(n)])
    for i in range(n):
        result[f"p{i}"] = ret[group == i].mean(axis=1)
    return result


def _cal_group_return_nb(ret: pd.DataFrame, group: pd.DataFrame, n: int):
    """ 600ms
    :param ret: pd.DataFrame (N x M)
    :param group: pd.DataFrame (N x M)
    :param n: int
    :return: pd.DataFrame (N x n)
    """
    return pd.DataFrame(_group_return(ret.values, group.values, n),
                        index=ret.index,
                        columns=[f"p{i}" for i in range(n)])


def group_return(factor: pd.DataFrame, ret: pd.DataFrame, n: int, period=None):
    """
    :param factor: pd.DataFrame, factor data (N x M)
    :param ret: pd.DataFrame, return data (N x M)
    :param n: int, number of group
    :return:
        ( pd.DataFrame, group return (N x n), pd.DataFrame, group tag (N x M))

    performance: 3.33 s ± 18.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """
    group = tag_group_nb(factor, n)
    if period is None:
        return {"1D": _cal_group_return_nb(ret, group, n)}, group
    else:
        result = {}
        for p in period:
            result[f"{p}D"] = _cal_group_return_nb(ret.rolling(p).sum(), group, n)
        return result, group
