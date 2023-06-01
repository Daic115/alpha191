# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : ic.py
# Time       ：2022/9/19 10:40
"""
import numpy as np
import numba as nb
import pandas as pd
from scipy import stats


@nb.jit(nopython=True, nogil=True)
def pearsonr(A: np.ndarray, B: np.ndarray):
    """
    计算pearsonr相关系数
    :param A: np.ndarray (N x M)
    :param B: np.ndarray (N x M)
    :return: np.ndarray (N)
    """
    N, M = A.shape
    result = np.zeros(N, dtype=np.float64)
    for i in range(N):
        a = A[i, :]
        b = B[i, :]
        result[i] = stats.pearsonr(a, b)[0]
    return result


@nb.jit(nopython=True, nogil=True)
def spearman(A: np.ndarray, B: np.ndarray):
    """
    计算Spearman相关系数
    :param A: (N, T)
    :param B: (N, T)
    :return: (N,)
    """
    N, T = A.shape
    corr = np.empty(N)
    for i in range(N):
        x = A[i, :]
        y = B[i, :]
        std = np.std(x) * np.std(y)
    return corr


def ic(factors: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """
    计算因子IC
    :param factors: 因子值 (N, T)
    :param returns: 收益率 (N, T)
    :return: IC
    """
    # 计算IC
    ic = factors.corrwith(returns, axis=1, method='pearson')
    return ic


def ic_rank(factors: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """
    计算因子RANK IC
    :param factors: 因子值 (N, T)
    :param returns: 收益率 (N, T)
    :return: IC
    """
    # 计算IC
    ic = factors.corrwith(returns, axis=1, method='spearman')
    return ic


def ic_quantile():
    # TODO
    pass


def ic_ir():
    pass
