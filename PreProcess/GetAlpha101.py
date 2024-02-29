from jqdatasdk import *
import pandas as pd
import os
from jqdatasdk.alpha101 import *

import datetime

'''
1. 获取 alpha101 指标，与股票的OHLC数据拼接，存成 processed_alpha.csv
2. 股价（delayed 1）和 alpha101 全指标计算互相关，生成互相关热力图，按相关性选取最优的20个因子
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

for stock_code in available_stock_list:
    target_dir = f"Data/{stock_code}"
    processed_file = f"{target_dir}/data_processed.csv"
    alpha101_file = f"{target_dir}/alpha101.csv"
    df = pd.read_csv(processed_file)
    alpha_df = pd.DataFrame()
    for date in df.Date:
        alpha_df = alpha_df.append(get_all_alpha_101(code=normalize_code(stock_code), date=date))
    alpha_df = alpha_df.fillna(value=0)
    pd.concat([df, alpha_df.reset_index(drop=True)], axis=1).to_csv(alpha101_file)
