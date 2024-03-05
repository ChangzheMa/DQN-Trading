import collections
import math

import numpy as np
from jqdatasdk import *
import pandas as pd
import os
from jqdatasdk.alpha101 import *
from sklearn.feature_selection import mutual_info_regression

import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

'''
计算指标与收盘价之间的关系，取最优的10个
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


def feature_importances_with_continuous_label(df, label_column, feature_columns):
    """
    将连续的标签转换为10个类别，使用随机森林分类器计算并返回特征的重要性。

    :param df: 输入的DataFrame。
    :param label_column: 连续标签列的名称。
    :param feature_columns: 特征列名的列表。
    :return: 一个字典，包含每个特征及其对应的重要性。
    """
    # 将连续标签转换为10个类别
    df['categorized_label'] = pd.qcut(df[label_column], q=10, labels=False, duplicates='drop')

    X = df[feature_columns]
    y = df['categorized_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用随机森林分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    feature_importances = [(feature, importance) for feature, importance in
                           zip(feature_columns, clf.feature_importances_)]

    return feature_importances


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
        print(f"test for {stock_code}, lag {lag}")
        label_name = f"close_return_lag{lag}"
        feature_name_list = [f"alpha_{('000' + str(index))[-3:]}" for index in range(1, 102)]
        scores[stock_code][f"lag{lag}"] = feature_importances_with_continuous_label(data, label_name, feature_name_list)
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
