import collections
import math

import numpy as np
from jqdatasdk import *
import pandas as pd
import os
from jqdatasdk.alpha101 import *
from sklearn.feature_selection import mutual_info_regression

import datetime

'''
计算指标与收盘价之间的互信息，取最优的10个
'''

auth('18612981484', 'Mcz19890521...')
print(get_query_count())

available_stock_list = [
    '000651',
    '000725',
    '000858',
    '600030',
    '600036',
    '600276',
    '600519',
    '600887',
    '600900',
    '601398'
]


def sort_by_score(item):
    return abs(item[1]) if not np.isnan(item[1]) else 0


scores = {}

lag_list = [1, 2, 3, 5, 10, 20, 50]

for stock_code in available_stock_list:
    data = pd.read_csv(f"../Data/{stock_code}/alpha101.csv")
    scores[stock_code] = {}

    for lag in lag_list:
        data[f"close_return_lag{lag}"] = data['close_norm'].shift(-lag) / data['close_norm']

    # 处理 inf -inf
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 处理 过大/过小 的值
    numerical_data = data.select_dtypes(include=[np.number])
    numerical_data = numerical_data.where((numerical_data <= 1e6) & (numerical_data >= 1e-6), np.nan)
    data.update(numerical_data)
    # 把所有不合法的值丢掉
    data.dropna(inplace=True)

    for lag in lag_list:
        scores[stock_code][f"lag{lag}"] = []
        for index in range(1, 102):
            # 计算互信息分数
            col_name = f"alpha_{('000' + str(index))[-3:]}"
            # score = mutual_info_regression(data.loc[:, col_name:col_name], data[f"close_return_lag{lag}"])  # 互信息
            # score = data[col_name].corr(data[f"close_return_lag{lag}"], method='spearman')  # spearman 相关系数
            score = data[col_name].corr(data[f"close_return_lag{lag}"])  # pearson 相关系数
            scores[stock_code][f"lag{lag}"].append((col_name, score))

        scores[stock_code][f"lag{lag}"].sort(key=sort_by_score, reverse=True)

col_names = []
for stock_code in scores.keys():
    for lag_name in scores[stock_code].keys():
        col_names.extend([item for (item, score) in scores[stock_code][lag_name][:50]])

for item, count in collections.Counter(col_names).most_common():
    print(f"'{item}',")

'''
Counter({'alpha_019': 10,
         'alpha_026': 10,
         'alpha_024': 10,
         'alpha_074': 10,
         'alpha_081': 10,
         'alpha_032': 9,
         'alpha_050': 9,
         'alpha_099': 9,
         'alpha_088': 8,
         'alpha_061': 8,
         'alpha_060': 7,
         'alpha_095': 7,
         'alpha_068': 5,
         'alpha_064': 5,
         'alpha_065': 4,
         'alpha_075': 4,
         'alpha_028': 3,
         'alpha_052': 3,
         'alpha_040': 3,
         'alpha_086': 3,
         'alpha_042': 2,
         'alpha_071': 1,
         'alpha_077': 1,
         'alpha_022': 1,
         'alpha_047': 1,
         'alpha_094': 1,
         'alpha_041': 1,
         'alpha_025': 1,
         'alpha_083': 1,
         'alpha_037': 1,
         'alpha_012': 1,
         'alpha_009': 1})


'''
