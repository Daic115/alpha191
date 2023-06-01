# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : __init__.py.py
# Time       ：2022/8/23 18:40
"""
import numpy as np
import pandas as pd
from scipy import stats
from .group_return import group_return
from .ic import ic, ic_rank, ic_ir

from typing import Union, List, Dict, Tuple, Optional


class FactorAnalysis(object):
    def __init__(self, save_img=False):
        self.save_img = save_img

    def analysis(self, factors, returns, n=5, period=None) -> dict:
        """
        :param factors: pd.DataFrame, index is datetime, columns is asset
        :param returns: pd.DataFrame, index is datetime, columns is asset,
                        recommend to use: next_close/next_open - 1
        :param n: int, number of quantile
        :param period: analysis period default is [1,3,5]
        """
        self.results = {}
        if period is None:
            self.period = [1, 3, 5]
        res_ic = ic(factors, returns)
        res_ic_rank = ic_rank(factors, returns)
        greturn, gtag = group_return(factors, returns, n, period=self.period)
        self._add_ic_stats(res_ic, tag="")
        self._add_ic_stats(res_ic_rank, tag="Rank ")
        for p in self.period:
            self._add_group_stats(greturn[f"{p}D"], n, p)
        if self.save_img:
            self._add_group_return_plot(greturn, gtag, n)
        return self.results

    def _add_ic_stats(self, ic_data: pd.Series, tag: str = ""):
        self.results[f"{tag}IC Mean"] = ic_data.mean()
        self.results[f"{tag}IC Std."] = ic_data.std()

        self.results[f"{tag}IC IR"] = \
            ic_data.mean() / ic_data.std()
        t_stat, p_value = stats.ttest_1samp(ic_data, 0, nan_policy='omit')
        self.results[f"{tag}IC t-stat"] = t_stat
        self.results[f"{tag}IC p-value"] = p_value
        # self.results[f"{tag}IC Skew"] = stats.skew(ic_data, nan_policy='omit')
        # self.results[f"{tag}IC Kurtosis"] = stats.kurtosis(ic_data, nan_policy='omit')

    def _add_group_stats(self, group_return, n, p):
        group_return = group_return.fillna(0.)
        # self.results[f"Cumsum Return {p}D"] = str(group_return.cumsum(axis=0).iloc[-1].tolist())
        longshort = group_return[f"p{n - 1}"] - group_return["p0"]
        self.results[f"Long-short Return {p}D"] = longshort.cumsum().iloc[-1]
        self.results[f"Long-short Sharpe {p}D"] = longshort.mean() / (longshort.std() + 1e-7)
        self.results[f"Min-Max Sharpe Diff {p}D"] = \
            (group_return[f"p{n - 1}"].mean() / group_return[f"p{n - 1}"].std() + 1e-7) - \
            (group_return[f"p0"].mean() / group_return[f"p0"].std() + 1e-7)

    def _add_group_return_plot(self, group_returns: Dict, group_tag: pd.DataFrame, n: int):
        pass
