import collections
import math

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
    return item[1]


scores = {}

for stock_code in available_stock_list:
    data = pd.read_csv(f"Data/{stock_code}/alpha101.csv")
    scores[stock_code] = []
    for index in range(1, 102):
        # 价格
        data['close_norm_lag1'] = data['close_norm'].shift(-1)
        data.dropna(inplace=True)
        # 技术指标
        col_name = f"alpha_{('000' + str(index))[-3:]}"
        score = mutual_info_regression(data.loc[:, col_name:col_name], data['close_norm_lag1'])
        scores[stock_code].append((col_name, score))

    scores[stock_code].sort(key=sort_by_score, reverse=True)

col_names = []
for key in scores.keys():
    col_names.extend([item for (item, score) in scores[key][:15]])

collections.Counter(col_names)

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
