# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : rolling.py
# Time       : 2022/9/17 13:42
# Author     : Daic
"""
import numpy as np
import numba as nb
import pandas as pd


# @nb.jit("f8[:](f8[:,:], f8[:,:], i8)")
@nb.jit(nopython=True, nogil=True)
def _pair_slope_rolling(A_arr: np.ndarray, B_arr: np.ndarray, n: int):
    """
    两个矩阵的滚动回归斜率
    其中A的每一列对应B的每一列
    A_arr: (N, L)
    B_arr: (N, L)
    """
    N, L = A_arr.shape
    result = np.zeros((N, L))
    result[:n] = np.nan
    for rolling_ind in range(n - 1, A_arr.shape[0]):
        bch_x = A_arr[rolling_ind - n + 1: rolling_ind + 1, :]
        bch_y = B_arr[rolling_ind - n + 1: rolling_ind + 1, :]
        for col_ind in range(L):
            x = bch_x[:, col_ind]  # (n, )
            y = bch_y[:, col_ind]  # (n, )
            x_mean = x - np.mean(x)
            y_mean = y - np.mean(y)
            result[rolling_ind, col_ind] = np.sum(x_mean * y_mean) / np.sum(x_mean ** 2)
    return result


@nb.jit(nopython=True, nogil=True)
def _pair_resi_rolling(A_arr: np.ndarray, B_arr: np.ndarray, n: int):
    """
    两个矩阵的滚动回归残差
    其中A的每一列对应B的每一列
    A_arr: (N, L)
    B_arr: (N, L)
    """
    N, L = A_arr.shape
    result = np.zeros((N, L))
    result[:n] = np.nan
    for rolling_ind in range(n - 1, A_arr.shape[0]):
        bch_x = A_arr[rolling_ind - n + 1: rolling_ind + 1, :]
        bch_y = B_arr[rolling_ind - n + 1: rolling_ind + 1, :]
        for col_ind in range(L):
            x = bch_x[:, col_ind]
            y = bch_y[:, col_ind]
            x_mean = x - np.mean(x)
            y_mean = y - np.mean(y)
            slope = np.sum(x_mean * y_mean) / np.sum(x_mean ** 2)
            result[rolling_ind, col_ind] = np.sum(y_mean - slope * x_mean)


@nb.jit(nopython=True, nogil=True)
def _wma(A_arr: np.ndarray, weight: np.ndarray):
    """
    其中A的每一列对应B的每一列
    A_arr: (N, L)
    B_arr: (N, L)
    """
    N, L = A_arr.shape
    n = weight.shape[0]
    result = np.zeros((N, L))
    result[:n] = np.nan
    for rolling_ind in range(n - 1, A_arr.shape[0]):
        bch_x = A_arr[rolling_ind - n + 1: rolling_ind + 1, :]
        for col_ind in range(L):
            x = bch_x[:, col_ind]
            if np.all(np.isnan(x)):
                result[rolling_ind, col_ind] = np.nan
            else:
                result[rolling_ind, col_ind] = np.nansum(x * weight)
        # result[rolling_ind, :] = np.dot(weight, bch_x)
    return result


@nb.jit(nopython=True, nogil=True)
def _alpha191_143(A_arr: np.ndarray, A_delay: np.ndarray):
    """
    CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF
    """
    N, L = A_arr.shape
    result = np.zeros((N, L))
    result[:1] = np.nan
    self_arr = np.ones(L, dtype=np.float64)
    for i in range(1, N):
        for stk in range(L):
            c = A_arr[i, stk]
            c_delay = A_delay[i, stk]
            if np.isnan(c) or np.isnan(c_delay):
                result[i, stk] = np.nan
                continue
            if A_arr[i, stk] > A_delay[i, stk]:
                result[i, stk] = (A_arr[i, stk] - A_delay[i, stk]) / A_delay[i, stk] * self_arr[stk]
                self_arr[stk] = result[i, stk]
            else:
                result[i, stk] = self_arr[stk]
    return result


def _min_distance(A_arr: np.ndarray, n: int):
    """
    其中A的每一列对应B的每一列
    A_arr: (N, L)
    B_arr: (N, L)
    """
    N, L = A_arr.shape
    result = np.zeros((N, L))
    result[:n] = np.nan
    for rolling_ind in range(n - 1, A_arr.shape[0]):
        bch_x = A_arr[rolling_ind - n + 1: rolling_ind + 1, :]
        max_ind = np.argmin(bch_x, axis=0)
        result[rolling_ind, :] = n - max_ind
    return result


def _max_distance(A_arr: np.ndarray, n: int):
    """
    其中A的每一列对应B的每一列
    A_arr: (N, L)
    B_arr: (N, L)
    """
    N, L = A_arr.shape
    result = np.zeros((N, L))
    result[:n] = np.nan
    for rolling_ind in range(n - 1, A_arr.shape[0]):
        bch_x = A_arr[rolling_ind - n + 1: rolling_ind + 1, :]
        max_ind = np.argmax(bch_x, axis=0)
        result[rolling_ind, :] = n - max_ind
    return result


def rolling_slope_pair(A: pd.DataFrame, B: pd.DataFrame, n: int):
    assert A.shape == B.shape
    slope = _pair_slope_rolling(A.values, B.values, n)
    return pd.DataFrame(slope, index=A.index, columns=A.columns)


if __name__ == '__main__':
    import time

    X = np.random.randn(1500, 5000)
    Y = np.random.randn(1500, 5000)

    # start = time.time()
    # for i in range(10):
    #     _ = _pair_slope_rolling0(X, Y, 16)
    # print(f"pair_slope_rolling0: {(time.time() - start) * 100} ms")
    #
    # start = time.time()
    # for i in range(10):
    #     _ = _pair_slope_rolling1(X, Y, 16)
    # print(f"pair_slope_rolling0: {(time.time() - start) * 100} ms")

    start = time.time()
    for i in range(5):
        print(i)
        _ = _pair_slope_rolling2(X, Y, np.int64(16))
    print(f"pair_slope_rolling0: {(time.time() - start) * 100} ms")
