from jqdatasdk import *
import pandas as pd
import os

import datetime

from DataLoader.DataLoader import YahooFinanceDataLoader

auth('18612981484', 'Mcz19890521...')
print(get_query_count())

YEAR_FILTER = 2007  # 发行日期必须早于此年
DATA_COUNT = 4000   # 取4000行数据
STOCK_LIST = [
    '600519.贵州茅台',
    '601318.中国平安',
    '300750.宁德时代',
    '600036.招商银行',
    '000333.美的集团',
    '000858.五粮液',
    '600900.长江电力',
    '601166.兴业银行',
    '600030.中信证券',
    '601899.紫金矿业',
    '601398.工商银行',
    '600887.伊利股份',
    '600276.恒瑞医药',
    '601328.交通银行',
    '300760.迈瑞医疗',
    '300059.东方财富',
    '000651.格力电器',
    '002594.比亚迪',
    '000725.京东方Ａ',
    '600919.江苏银行',
]

norm_stock_list = normalize_code(STOCK_LIST)

stock_info = [get_security_info(item) for item in norm_stock_list]

available_stock_info = [item for item in stock_info if item.start_date < datetime.date(YEAR_FILTER, 1, 1)]

for item in available_stock_info:
    print(item.display_name + str(item.start_date))

available_stock_list = [item.code for item in available_stock_info]

day_data = get_bars(available_stock_list, DATA_COUNT, unit='1d',
                    fields=['date', 'open', 'close', 'high', 'low', 'volume', 'money', 'factor'],
                    fq_ref_date='2024-01-01', df=True)

grouped = day_data.groupby(level=0)

for stock_code, group in grouped:
    target_dir = f"Data/{stock_code}"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    filename = f"{target_dir}/{stock_code}.csv"
    group.rename(columns={
        'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'date': 'Date',
        'volume': 'Volume', 'money': 'Money', 'factor': 'Factor'
    }, inplace=True)
    group['Adj Close'] = group['Close']
    group.to_csv(filename)
    data_loader = YahooFinanceDataLoader(dataset_name=stock_code, split_point="2020-01-01")
    data_loader.plot_data()
