from jqdatasdk import *
import pandas as pd
import os

import datetime

from DataLoader.DataLoader import YahooFinanceDataLoader

'''
股票列表：
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

'''

stock_list = [
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
DATA_COUNT = 4000   # 取4000行数据

auth('18612981484', 'Mcz19890521...')
print(get_query_count())

week_data = get_bars(normalize_code(stock_list), DATA_COUNT, unit='1w',
                    fields=['date', 'open', 'close', 'high', 'low', 'volume', 'money', 'factor'],
                    fq_ref_date='2024-01-01', df=True)
