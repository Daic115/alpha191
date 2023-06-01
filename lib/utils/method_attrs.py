# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : method_attrs.py
# Time       ：2022/9/20 16:53
"""


def permission(permission_required):
    def decorator(func):
        func.permission_required = permission_required
        return func

    return decorator


def factor_attr(max_depend=100, return_type="float", unfinished=False, **attrs):
    def decorator(func):
        func.max_depend = max_depend
        func.return_type = return_type
        func.unfinished = unfinished
        func.attrs = attrs
        return func

    return decorator