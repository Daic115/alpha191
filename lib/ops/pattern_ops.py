# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : pattern_ops.py
# Time       ：2022/9/21 9:51
"""
import numpy as np
import pandas as pd
import talib as ta
from typing import List, Union, Dict, Any


def merge_ta_bars(data: Dict[str, pd.DataFrame]):
    return pd.concat([data["open"], data["high"], data["low"], data["close"]], axis=0)


def _get_pattern(bar_arr, bar_func, *func_args, **func_kwargs):
    bar_arr = bar_arr.reshape(4, len(bar_arr) // 4)  # fix
    return bar_func(bar_arr[0, :], bar_arr[1, :], bar_arr[2, :], bar_arr[3, :], *func_args, **func_kwargs)


class TaPatternWrapper:
    def __init__(self, ta_func, *args, **kwargs):
        self.ta_func = ta_func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, bars: pd.DataFrame, keep_nan: bool = False):
        idx = bars.index[: len(bars) // 4]
        res = bars.apply(lambda x: _get_pattern(x.values, self.ta_func, *self.args, **self.kwargs))
        res.index = idx
        if keep_nan:
            nan_map = bars.iloc[:len(bars) // 4].isna().astype(float).replace(1., np.nan)
            res = res + nan_map
        return res


def TA_2CROWS(bars: pd.DataFrame, keep_nan: bool = False):
    """
    两只乌鸦
    三日K线模式，第一天长阳，第二天高开收阴，第三天再次高开继续收阴，收盘比前一日收盘价低
    """
    return TaPatternWrapper(ta.CDL2CROWS)(bars, keep_nan)


def TA_3BLACKCROWS(bars: pd.DataFrame, keep_nan: bool = False):
    """
    三只乌鸦
    三日K线模式，连续三根阴线，每日收盘价都下跌且接近最低价，每日开盘价都在上根K线实体内
    """
    return TaPatternWrapper(ta.CDL3BLACKCROWS)(bars, keep_nan)


def TA_3INSIDE(bars: pd.DataFrame, keep_nan: bool = False):
    """
    三日K线模式，母子信号+长K线，以三内部上涨为例，K线为阴阳阳，第三天收盘价高于第一天开盘价，第二天K线在第一天K线内部，预示着股价上涨。
    """
    return TaPatternWrapper(ta.CDL3INSIDE)(bars, keep_nan)


def TA_3LINESTRIKE(bars: pd.DataFrame, keep_nan: bool = False):
    """
    三线打击
    四日K线模式，前三根阳线，每日收盘价都比前一日高，开盘价在前一日实体内，第四日市场高开，收盘价低于第一日开盘价，预示股价下跌。
    """
    return TaPatternWrapper(ta.CDL3LINESTRIKE)(bars, keep_nan)


def TA_3OUTSIDE(bars: pd.DataFrame, keep_nan: bool = False):
    """
    三外部上涨和下跌
    三日K线模式，与三内部上涨和下跌类似，K线为阴阳阳，但第一日与第二日的K线形态相反，以三外部上涨为例，第一日K线在第二日K线内部，预示着股价上涨。
    """
    return TaPatternWrapper(ta.CDL3OUTSIDE)(bars, keep_nan)


def TA_3STARSINSOUTH(bars: pd.DataFrame, keep_nan: bool = False):
    """
    南方三星
    三日K线模式，与大敌当前相反，三日K线皆阴，第一日有长下影线，第二日与第一日类似，K线整体小于第一日，
    第三日无下影线实体信号，成交价格都在第一日振幅之内，预示下跌趋势反转，股价上升
    """
    return TaPatternWrapper(ta.CDL3STARSINSOUTH)(bars, keep_nan)


def TA_3WHITESOLDIERS(bars: pd.DataFrame, keep_nan: bool = False):
    """
    三个白兵
    三日K线模式，三日K线皆阳，每日收盘价变高且接近最高价，开盘价在前一日实体上半部，预示股价上升
    """
    return TaPatternWrapper(ta.CDL3WHITESOLDIERS)(bars, keep_nan)


def TA_ABANDONEDBABY(bars: pd.DataFrame, keep_nan: bool = False, penetration=0.3):
    """
    弃婴
    三日K线模式，第二日价格跳空且收十字星（开盘价与收盘价接近，最高价最低价相差不大），预示趋势反转，发生在顶部下跌，底部上涨
    """
    return TaPatternWrapper(ta.CDLABANDONEDBABY, penetration=penetration)(bars, keep_nan)


def TA_ADVANCEBLOCK(bars: pd.DataFrame, keep_nan: bool = False):
    """
    大敌当前
    三日K线模式，三日都收阳，每日收盘价都比前一日高，开盘价都在前一日实体以内，实体变短，上影线变长
    """
    return TaPatternWrapper(ta.CDLADVANCEBLOCK)(bars, keep_nan)


def TA_BELTHOLD(bars: pd.DataFrame, keep_nan: bool = False):
    """
    捉腰带线
    两日K线模式，下跌趋势中，第一日阴线，第二日开盘价为最低价，阳线，收盘价接近最高价，预示价格上涨。
    """
    return TaPatternWrapper(ta.CDLBELTHOLD)(bars, keep_nan)


def TA_BREAKAWAY(bars: pd.DataFrame, keep_nan: bool = False):
    """
    脱离
    五日K线模式，以看涨脱离为例，下跌趋势中，第一日长阴线，第二日跳空阴线，延续趋势开始震荡，第五日长阳线，
    收盘价在第一天收盘价与第二天开盘价之间，预示价格上涨
    """
    return TaPatternWrapper(ta.CDLBREAKAWAY)(bars, keep_nan)


def TA_CLOSINGMARUBOZU(bars: pd.DataFrame, keep_nan: bool = False):
    """
    收盘缺影线
    一日K线模式，以阳线为例，最低价低于开盘价，收盘价等于最高价，预示着趋势持续
    """
    return TaPatternWrapper(ta.CDLCLOSINGMARUBOZU)(bars, keep_nan)


def TA_CONCEALBABYSWALL(bars: pd.DataFrame, keep_nan: bool = False):
    """
    藏婴吞没
    四日K线模式，下跌趋势中，前两日阴线无影线，第二日开盘、收盘价皆低于第二日，第三日倒锤头，第四日开盘价高于前一日最高价，收盘价低于前一日最低价，预示着底部反转
    """
    return TaPatternWrapper(ta.CDLCONCEALBABYSWALL)(bars, keep_nan)


def TA_COUNTERATTACK(bars: pd.DataFrame, keep_nan: bool = False):
    """
    反击线
    二日K线模式，与分离线类似 ，预示着趋势继续
    """
    return TaPatternWrapper(ta.CDLCOUNTERATTACK)(bars, keep_nan)


def TA_DARKCLOUDCOVER(bars: pd.DataFrame, keep_nan: bool = False, penetration=0.5):
    """
    乌云压顶
    二日K线模式，第一日长阳，第二日开盘价高于前一日最高价，收盘价处于前一日实体中部以下，预示着股价下跌
    """
    return TaPatternWrapper(ta.CDLDARKCLOUDCOVER, penetration=penetration)(bars, keep_nan)


def TA_DOJI(bars: pd.DataFrame, keep_nan: bool = False):
    """
    十字
    一日K线模式，开盘价与收盘价基本相同
    """
    return TaPatternWrapper(ta.CDLDOJI)(bars, keep_nan)


def TA_DOJISTAR(bars: pd.DataFrame, keep_nan: bool = False):
    """
    十字星
    一日K线模式，开盘价与收盘价基本相同，上下影线不会很长，预示着当前趋势反转
    """
    return TaPatternWrapper(ta.CDLDOJISTAR)(bars, keep_nan)


def TA_DRAGONFLYDOJI(bars: pd.DataFrame, keep_nan: bool = False):
    """
    蜻蜓十字/T形十字
    一日K线模式，开盘后价格一路走低，之后收复，收盘价与开盘价相同，预示趋势反转
    """
    return TaPatternWrapper(ta.CDLDRAGONFLYDOJI)(bars, keep_nan)


def TA_ENGULFING(bars: pd.DataFrame, keep_nan: bool = False):
    """
    吞噬模式
    两日K线模式，分多头吞噬和空头吞噬，以多头吞噬为例，第一日为阴线，第二日阳线，第一日的开盘价和收盘价在第二日开盘价收盘价之内，但不能完全相同
    """
    return TaPatternWrapper(ta.CDLENGULFING)(bars, keep_nan)


def TA_EVENINGDOJISTAR(bars: pd.DataFrame, keep_nan: bool = False, penetration=0.3):
    """
    十字暮星
    三日K线模式，基本模式为暮星，第二日收盘价和开盘价相同，预示顶部反转
    """
    return TaPatternWrapper(ta.CDLEVENINGDOJISTAR, penetration=penetration)(bars, keep_nan)


def TA_EVENINGSTAR(bars: pd.DataFrame, keep_nan: bool = False, penetration=0.3):
    """
    暮星
    三日K线模式，与晨星相反，上升趋势中,第一日阳线，第二日价格振幅较小，第三日阴线，预示顶部反转
    """
    return TaPatternWrapper(ta.CDLEVENINGSTAR, penetration=penetration)(bars, keep_nan)


def TA_GAPSIDESIDEWHITE(bars: pd.DataFrame, keep_nan: bool = False):
    """
    向上/下跳空并列阳线
    二日K线模式，上升趋势向上跳空，下跌趋势向下跳空,第一日与第二日有相同开盘价，实体长度差不多，则趋势持续
    """
    return TaPatternWrapper(ta.CDLGAPSIDESIDEWHITE)(bars, keep_nan)


def TA_GRAVESTONEDOJI(bars: pd.DataFrame, keep_nan: bool = False):
    """
    墓碑十字/倒T十字
    一日K线模式，开盘价与收盘价相同，上影线长，无下影线，预示底部反转
    """
    return TaPatternWrapper(ta.CDLGRAVESTONEDOJI)(bars, keep_nan)


def TA_HAMMER(bars: pd.DataFrame, keep_nan: bool = False):
    """
    锤头
    一日K线模式，实体较短，无上影线，下影线大于实体长度两倍，处于下跌趋势底部，预示反转
    """
    return TaPatternWrapper(ta.CDLHAMMER)(bars, keep_nan)


def TA_HANGINGMAN(bars: pd.DataFrame, keep_nan: bool = False):
    """
    上吊线
    一日K线模式，形状与锤子类似，处于上升趋势的顶部，预示着趋势反转
    """
    return TaPatternWrapper(ta.CDLHANGINGMAN)(bars, keep_nan)


def TA_HARAMI(bars: pd.DataFrame, keep_nan: bool = False):
    """
    母子线
    二日K线模式，分多头母子与空头母子，两者相反，以多头母子为例，在下跌趋势中，第一日K线长阴，第二日开盘价收盘价在第一日价格振幅之内，为阳线，预示趋势反转，股价上升
    """
    return TaPatternWrapper(ta.CDLHARAMI)(bars, keep_nan)


def TA_HARAMICROSS(bars: pd.DataFrame, keep_nan: bool = False):
    """
    十字孕线
    二日K线模式，与母子县类似，若第二日K线是十字线，便称为十字孕线，预示着趋势反转
    """
    return TaPatternWrapper(ta.CDLHARAMICROSS)(bars, keep_nan)


def TA_HIGHWAVE(bars: pd.DataFrame, keep_nan: bool = False):
    """
    风高浪大线
    三日K线模式，具有极长的上/下影线与短的实体，预示着趋势反转
    """
    return TaPatternWrapper(ta.CDLHIGHWAVE)(bars, keep_nan)


def TA_HIKKAKE(bars: pd.DataFrame, keep_nan: bool = False):
    """
    陷阱
    三日K线模式，与母子类似，第二日价格在前一日实体范围内,第三日收盘价高于前两日，反转失败，趋势继续
    """
    return TaPatternWrapper(ta.CDLHIKKAKE)(bars, keep_nan)


def TA_HIKKAKEMOD(bars: pd.DataFrame, keep_nan: bool = False):
    """
    修正陷阱
    三日K线模式，与陷阱类似，上升趋势中，第三日跳空高开；下跌趋势中，第三日跳空低开，反转失败，趋势继续
    """
    return TaPatternWrapper(ta.CDLHIKKAKEMOD)(bars, keep_nan)


def TA_HOMINGPIGEON(bars: pd.DataFrame, keep_nan: bool = False):
    """
    家鸽
    二日K线模式，与母子线类似，不同的的是二日K线颜色相同，第二日最高价、最低价都在第一日实体之内，预示着趋势反转
    """
    return TaPatternWrapper(ta.CDLHOMINGPIGEON)(bars, keep_nan)


def TA_IDENTICAL3CROWS(bars: pd.DataFrame, keep_nan: bool = False):
    """
    三胞胎乌鸦
    三日K线模式，上涨趋势中，三日都为阴线，长度大致相等，每日开盘价等于前一日收盘价，收盘价接近当日最低价，预示价格下跌
    """
    return TaPatternWrapper(ta.CDLIDENTICAL3CROWS)(bars, keep_nan)


def TA_INNECK(bars: pd.DataFrame, keep_nan: bool = False):
    """
    颈内线
    二日K线模式，下跌趋势中，第一日长阴线，第二日开盘价较低，收盘价略高于第一日收盘价，阳线，实体较短，预示着下跌继续
    """
    return TaPatternWrapper(ta.CDLINNECK)(bars, keep_nan)


def TA_INVERTEDHAMMER(bars: pd.DataFrame, keep_nan: bool = False):
    """
    倒锤头
    一日K线模式，上影线较长，长度为实体2倍以上，无下影线，在下跌趋势底部，预示着趋势反转
    """
    return TaPatternWrapper(ta.CDLINVERTEDHAMMER)(bars, keep_nan)


def TA_KICKING(bars: pd.DataFrame, keep_nan: bool = False):
    """
    反冲形态
    二日K线模式，与分离线类似，两日K线为秃线，颜色相反，存在跳空缺口
    """
    return TaPatternWrapper(ta.CDLKICKING)(bars, keep_nan)


def TA_KICKINGBYLENGTH(bars: pd.DataFrame, keep_nan: bool = False):
    """
    由较长缺影线决定的反冲形态
    二日K线模式，与反冲形态类似，较长缺影线决定价格的涨跌
    """
    return TaPatternWrapper(ta.CDLKICKINGBYLENGTH)(bars, keep_nan)


def TA_LADDERBOTTOM(bars: pd.DataFrame, keep_nan: bool = False):
    """
    梯底
    五日K线模式，下跌趋势中，前三日阴线，开盘价与收盘价皆低于前一日开盘、收盘价，第四日倒锤头，第五日开盘价高于前一日开盘价，阳线，收盘价高于前几日价格振幅，预示着底部反转
    """
    return TaPatternWrapper(ta.CDLLADDERBOTTOM)(bars, keep_nan)


def TA_LONGLEGGEDDOJI(bars: pd.DataFrame, keep_nan: bool = False):
    """
    长脚十字
    一日K线模式，开盘价与收盘价相同居当日价格中部，上下影线长，表达市场不确定性
    """
    return TaPatternWrapper(ta.CDLLONGLEGGEDDOJI)(bars, keep_nan)


def TA_LONGLINE(bars: pd.DataFrame, keep_nan: bool = False):
    """
    长蜡烛
    一日K线模式，K线实体长，无上下影线
    """
    return TaPatternWrapper(ta.CDLLONGLINE)(bars, keep_nan)


def TA_MARUBOZU(bars: pd.DataFrame, keep_nan: bool = False):
    """
    光头光脚/缺影线
    一日K线模式，上下两头都没有影线的实体，阴线预示着熊市持续或者牛市反转，阳线相反
    """
    return TaPatternWrapper(ta.CDLMARUBOZU)(bars, keep_nan)


def TA_MATCHINGLOW(bars: pd.DataFrame, keep_nan: bool = False):
    """
    相同低价
    二日K线模式，下跌趋势中，第一日长阴线，第二日阴线，收盘价与前一日相同，预示底部确认，该价格为支撑位
    """
    return TaPatternWrapper(ta.CDLMATCHINGLOW)(bars, keep_nan)


def TA_MATHOLD(bars: pd.DataFrame, keep_nan: bool = False, penetration=0.5):
    """
    铺垫
    五日K线模式，上涨趋势中，第一日阳线，第二日跳空高开影线，第三、四日短实体影线，第五日阳线，收盘价高于前四日，预示趋势持续
    """
    return TaPatternWrapper(ta.CDLMATHOLD, penetration=penetration)(bars, keep_nan)


def TA_MORNINGDOJISTAR(bars: pd.DataFrame, keep_nan: bool = False, penetration=0.3):
    """
    十字晨星
    三日K线模式，基本模式为晨星，第二日K线为十字星，预示底部反转
    """
    return TaPatternWrapper(ta.CDLMORNINGDOJISTAR, penetration=penetration)(bars, keep_nan)


def TA_MORNINGSTAR(bars: pd.DataFrame, keep_nan: bool = False, penetration=0.3):
    """
    晨星
    三日K线模式，下跌趋势，第一日阴线，第二日价格振幅较小，第三天阳线，预示底部反转
    """
    return TaPatternWrapper(ta.CDLMORNINGSTAR, penetration=penetration)(bars, keep_nan)


def TA_ONNECK(bars: pd.DataFrame, keep_nan: bool = False):
    """
    颈上线
    二日K线模式，下跌趋势中，第一日长阴线，第二日开盘价较低，收盘价与前一日最低价相同，阳线，实体较短，预示着延续下跌趋势
    """
    return TaPatternWrapper(ta.CDLONNECK)(bars, keep_nan)


def TA_PIERCING(bars: pd.DataFrame, keep_nan: bool = False):
    """
    刺透形态
    两日K线模式，下跌趋势中，第一日阴线，第二日收盘价低于前一日最低价，收盘价处在第一日实体上部，预示着底部反转
    """
    return TaPatternWrapper(ta.CDLPIERCING)(bars, keep_nan)


def TA_RICKSHAWMAN(bars: pd.DataFrame, keep_nan: bool = False):
    """
    黄包车夫
    一日K线模式，与长腿十字线类似，若实体正好处于价格振幅中点，称为黄包车夫
    """
    return TaPatternWrapper(ta.CDLRICKSHAWMAN)(bars, keep_nan)


def TA_RISEFALL3METHODS(bars: pd.DataFrame, keep_nan: bool = False):
    """
    上升/下降三法
    五日K线模式，以上升三法为例，上涨趋势中，第一日长阳线，中间三日价格在第一日范围内小幅震荡，第五日长阳线，收盘价高于第一日收盘价，预示股价上升
    """
    return TaPatternWrapper(ta.CDLRISEFALL3METHODS)(bars, keep_nan)


def TA_SEPARATINGLINES(bars: pd.DataFrame, keep_nan: bool = False):
    """
    分离线
    二日K线模式，上涨趋势中，第一日阴线，第二日阳线，第二日开盘价与第一日相同且为最低价，预示着趋势继续
    """
    return TaPatternWrapper(ta.CDLSEPARATINGLINES)(bars, keep_nan)


def TA_SHOOTINGSTAR(bars: pd.DataFrame, keep_nan: bool = False):
    """
    射击之星
    一日K线模式，上影线至少为实体长度两倍，没有下影线，预示着股价下跌
    """
    return TaPatternWrapper(ta.CDLSHOOTINGSTAR)(bars, keep_nan)


def TA_SHORTLINE(bars: pd.DataFrame, keep_nan: bool = False):
    """
    短蜡烛
    一日K线模式，实体短，无上下影线
    """
    return TaPatternWrapper(ta.CDLSHORTLINE)(bars, keep_nan)


def TA_SPINNINGTOP(bars: pd.DataFrame, keep_nan: bool = False):
    """
    纺锤
    一日K线，实体小
    """
    return TaPatternWrapper(ta.CDLSPINNINGTOP)(bars, keep_nan)


def TA_STALLEDPATTERN(bars: pd.DataFrame, keep_nan: bool = False):
    """
    停顿形态
    三日K线模式，上涨趋势中，第二日长阳线，第三日开盘于前一日收盘价附近，短阳线，预示着上涨结束
    """
    return TaPatternWrapper(ta.CDLSTALLEDPATTERN)(bars, keep_nan)


def TA_STICKSANDWICH(bars: pd.DataFrame, keep_nan: bool = False):
    """
    条形三明治
    三日K线模式，第一日长阴线，第二日阳线，开盘价高于前一日收盘价，第三日开盘价高于前两日最高价，收盘价于第一日收盘价相同，反转
    """
    return TaPatternWrapper(ta.CDLSTICKSANDWICH)(bars, keep_nan)


def TA_TAKURI(bars: pd.DataFrame, keep_nan: bool = False):
    """
    探水竿
    一日K线模式，大致与蜻蜓十字相同，下影线长度长
    """
    return TaPatternWrapper(ta.CDLTAKURI)(bars, keep_nan)


def TA_TASUKIGAP(bars: pd.DataFrame, keep_nan: bool = False):
    """
    跳空并列阴阳线
    三日K线模式，分上涨和下跌，以上升为例，前两日阳线，第二日跳空，第三日阴线，收盘价于缺口中，上升趋势持续
    """
    return TaPatternWrapper(ta.CDLTASUKIGAP)(bars, keep_nan)


def TA_THRUSTING(bars: pd.DataFrame, keep_nan: bool = False):
    """
    插入
    二日K线模式，与颈上线类似，下跌趋势中，第一日长阴线，第二日开盘价跳空，收盘价略低于前一日实体中部，与颈上线相比实体较长，预示着趋势持续
    """
    return TaPatternWrapper(ta.CDLTHRUSTING)(bars, keep_nan)


def TA_TRISTAR(bars: pd.DataFrame, keep_nan: bool = False):
    """
    三星
    三日K线模式，由三个十字组成，第二日十字必须高于或者低于第一日和第三日，预示着反转
    """
    return TaPatternWrapper(ta.CDLTRISTAR)(bars, keep_nan)


def TA_UNIQUE3RIVER(bars: pd.DataFrame, keep_nan: bool = False):
    """
    奇特三河床
    三日K线模式，下跌趋势中，第一日长阴线，第二日为锤头，最低价创新低，第三日开盘价低于第二日收盘价，收阳线，收盘价不高于第二日收盘价，预示着反转，第二日下影线越长可能性越大
    """
    return TaPatternWrapper(ta.CDLUNIQUE3RIVER)(bars, keep_nan)


def TA_UPSIDEGAP2CROWS(bars: pd.DataFrame, keep_nan: bool = False):
    """
    向上跳空的两只乌鸦
    三日K线模式，第一日阳线，第二日跳空以高于第一日最高价开盘，收阴线，第三日开盘价高于第二日，收阴线，与第一日比仍有缺口
    """
    return TaPatternWrapper(ta.CDLUPSIDEGAP2CROWS)(bars, keep_nan)


def TA_XSIDEGAP3METHODS(bars: pd.DataFrame, keep_nan: bool = False):
    """
    上升/下降跳空三法
    五日K线模式，以上升跳空三法为例，上涨趋势中，第一日长阳线，第二日短阳线，第三日跳空阳线，第四日阴线，
    开盘价与收盘价于前两日实体内，第五日长阳线，收盘价高于第一日收盘价，预示股价上升
    """
    return TaPatternWrapper(ta.CDLXSIDEGAP3METHODS)(bars, keep_nan)


TA_PATTERN_FUNCS = {
    'CDL2CROWS': TA_2CROWS,
    'CDL3BLACKCROWS': TA_3BLACKCROWS,
    'CDL3INSIDE': TA_3INSIDE,
    'CDL3LINESTRIKE': TA_3LINESTRIKE,
    'CDL3OUTSIDE': TA_3OUTSIDE,
    'CDL3STARSINSOUTH': TA_3STARSINSOUTH,
    'CDL3WHITESOLDIERS': TA_3WHITESOLDIERS,
    'CDLABANDONEDBABY': TA_ABANDONEDBABY,
    'CDLADVANCEBLOCK': TA_ADVANCEBLOCK,
    'CDLBELTHOLD': TA_BELTHOLD,
    'CDLBREAKAWAY': TA_BREAKAWAY,
    'CDLCLOSINGMARUBOZU': TA_CLOSINGMARUBOZU,
    'CDLCONCEALBABYSWALL': TA_CONCEALBABYSWALL,
    'CDLCOUNTERATTACK': TA_COUNTERATTACK,
    'CDLDARKCLOUDCOVER': TA_DARKCLOUDCOVER,
    'CDLDOJI': TA_DOJI,
    'CDLDOJISTAR': TA_DOJISTAR,
    'CDLDRAGONFLYDOJI': TA_DRAGONFLYDOJI,
    'CDLENGULFING': TA_ENGULFING,
    'CDLEVENINGDOJISTAR': TA_EVENINGDOJISTAR,
    'CDLEVENINGSTAR': TA_EVENINGSTAR,
    'CDLGAPSIDESIDEWHITE': TA_GAPSIDESIDEWHITE,
    'CDLGRAVESTONEDOJI': TA_GRAVESTONEDOJI,
    'CDLHAMMER': TA_HAMMER,
    'CDLHANGINGMAN': TA_HANGINGMAN,
    'CDLHARAMI': TA_HARAMI,
    'CDLHARAMICROSS': TA_HARAMICROSS,
    'CDLHIGHWAVE': TA_HIGHWAVE,
    'CDLHIKKAKE': TA_HIKKAKE,
    'CDLHIKKAKEMOD': TA_HIKKAKEMOD,
    'CDLHOMINGPIGEON': TA_HOMINGPIGEON,
    'CDLIDENTICAL3CROWS': TA_IDENTICAL3CROWS,
    'CDLINNECK': TA_INNECK,
    'CDLINVERTEDHAMMER': TA_INVERTEDHAMMER,
    'CDLKICKING': TA_KICKING,
    'CDLKICKINGBYLENGTH': TA_KICKINGBYLENGTH,
    'CDLLADDERBOTTOM': TA_LADDERBOTTOM,
    'CDLLONGLEGGEDDOJI': TA_LONGLEGGEDDOJI,
    'CDLLONGLINE': TA_LONGLINE,
    'CDLMARUBOZU': TA_MARUBOZU,
    'CDLMATCHINGLOW': TA_MATCHINGLOW,
    'CDLMATHOLD': TA_MATHOLD,
    'CDLMORNINGDOJISTAR': TA_MORNINGDOJISTAR,
    'CDLMORNINGSTAR': TA_MORNINGSTAR,
    'CDLONNECK': TA_ONNECK,
    'CDLPIERCING': TA_PIERCING,
    'CDLRICKSHAWMAN': TA_RICKSHAWMAN,
    'CDLRISEFALL3METHODS': TA_RISEFALL3METHODS,
    'CDLSEPARATINGLINES': TA_SEPARATINGLINES,
    'CDLSHOOTINGSTAR': TA_SHOOTINGSTAR,
    'CDLSHORTLINE': TA_SHORTLINE,
    'CDLSPINNINGTOP': TA_SPINNINGTOP,
    'CDLSTALLEDPATTERN': TA_STALLEDPATTERN,
    'CDLSTICKSANDWICH': TA_STICKSANDWICH,
    'CDLTAKURI': TA_TAKURI,
    'CDLTASUKIGAP': TA_TASUKIGAP,
    'CDLTHRUSTING': TA_THRUSTING,
    'CDLTRISTAR': TA_TRISTAR,
    'CDLUNIQUE3RIVER': TA_UNIQUE3RIVER,
    'CDLUPSIDEGAP2CROWS': TA_UPSIDEGAP2CROWS,
    'CDLXSIDEGAP3METHODS': TA_XSIDEGAP3METHODS,
}
