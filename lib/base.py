# !/usr/bin/env python
# -*-coding:utf-8 -*-

from abc import ABCMeta, abstractmethod
from typing import List, Dict, Tuple, Union, Optional


class FactorBase(metaclass=ABCMeta):
    '''因子基础类
    '''

    def __init__(self, name: Union[List, str], desc: Union[List, str]):
        self.name = name
        self.desc = desc
        self.factor_data = {}

    @abstractmethod
    def cal(self, input_data=None, **kwargs):
        '''计算因子值
        '''
        pass

    def check_output(self):
        '''检查输出
        '''
        factors_names = self.factor_data.keys()
        if len(factors_names) == 0 or len(factors_names) != len(self.name):
            raise ValueError(f'因子计算结果为空或者因子名称与输出不一致, {self.name} != {factors_names}')

    def run(self, input_data=None) -> Dict:
        self.cal(input_data)
        self.check_output()
        return self.factor_data
