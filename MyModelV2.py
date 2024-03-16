# Importing DataLoaders for each model. These models include rule-based, vanilla DQN and encoder-decoder DQN.
import datetime
import math
import plotly.graph_objs as go

from DataLoader.DataLoader import YahooFinanceDataLoader
from DataLoader.DataForPatternBasedAgent import DataForPatternBasedAgent
from DataLoader.DataAutoPatternExtractionAgent import DataAutoPatternExtractionAgent
from DataLoader.DataSequential import DataSequential

from DeepRLAgent.MLPEncoder.Train import Train as SimpleMLP
from DeepRLAgent.MLPEncoderExtension.Train import Train as SimpleMLPExt
from DeepRLAgent.SimpleCNNEncoder.Train import Train as SimpleCNN
from EncoderDecoderAgent.GRU.Train import Train as GRU
from EncoderDecoderAgent.GRUExtension.Train import Train as GRUExt
from EncoderDecoderAgent.CNN.Train import Train as CNN
from EncoderDecoderAgent.CNN2D.Train import Train as CNN2d
from EncoderDecoderAgent.CNNAttn.Train import Train as CNN_ATTN
from EncoderDecoderAgent.CNN_GRU.Train import Train as CNN_GRU

# Imports for Deep RL Agent
from DeepRLAgent.VanillaInput.Train import Train as DeepRL

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import argparse
# from tqdm import tqdm
import os
import datetime as dt

from PatternDetectionInCandleStick.Evaluation import Evaluation
from utils import save_pkl, load_pkl
from types import SimpleNamespace
from torch.utils import tensorboard

tensorboard_writer = tensorboard.SummaryWriter('runs')

parser = argparse.ArgumentParser(description='DQN-Trader arguments')

parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument("--mode", default="client")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", default="63769")
args = parser.parse_args()

DATA_LOADERS = {
    'BTC-USD': YahooFinanceDataLoader('BTC-USD',
                                      split_point='2018-01-01',
                                      load_from_file=True),

    # 'GOOGL': YahooFinanceDataLoader('GOOGL',
    #                                 split_point='2018-01-01',
    #                                 load_from_file=True),
    #
    # 'AAPL': YahooFinanceDataLoader('AAPL',
    #                                split_point='2018-01-01',
    #                                begin_date='2010-01-01',
    #                                end_date='2020-08-24',
    #                                load_from_file=True),
    #
    # 'DJI': YahooFinanceDataLoader('DJI',
    #                               split_point='2016-01-01',
    #                               begin_date='2009-01-01',
    #                               end_date='2018-09-30',
    #                               load_from_file=True),
    #
    # 'S&P': YahooFinanceDataLoader('S&P',
    #                               split_point=2000,
    #                               end_date='2018-09-25',
    #                               load_from_file=True),
    #
    # 'AMD': YahooFinanceDataLoader('AMD',
    #                               split_point=2000,
    #                               end_date='2018-09-25',
    #                               load_from_file=True),
    #
    # 'GE': YahooFinanceDataLoader('GE',
    #                              split_point='2015-01-01',
    #                              load_from_file=True),
    #
    # 'KSS': YahooFinanceDataLoader('KSS',
    #                               split_point='2018-01-01',
    #                               load_from_file=True),
    #
    # 'HSI': YahooFinanceDataLoader('HSI',
    #                               split_point='2015-01-01',
    #                               load_from_file=True),
    #
    # 'AAL': YahooFinanceDataLoader('AAL',
    #                               split_point='2018-01-01',
    #                               load_from_file=True),

    '000651': YahooFinanceDataLoader('000651',
                                     split_point='2020-01-01',
                                     load_from_file=True),

    '000725': YahooFinanceDataLoader('000725',
                                     split_point='2020-01-01',
                                     load_from_file=True),

    '000858': YahooFinanceDataLoader('000858',
                                     split_point='2020-01-01',
                                     load_from_file=True),

    '600030': YahooFinanceDataLoader('600030',
                                     split_point='2020-01-01',
                                     load_from_file=True),

    '600036': YahooFinanceDataLoader('600036',
                                     split_point='2020-01-01',
                                     load_from_file=True),

    '600276': YahooFinanceDataLoader('600276',
                                     split_point='2020-01-01',
                                     load_from_file=True),

    '600519': YahooFinanceDataLoader('600519',
                                     split_point='2020-01-01',
                                     load_from_file=True),

    '600887': YahooFinanceDataLoader('600887',
                                     split_point='2020-01-01',
                                     load_from_file=True),

    '600900': YahooFinanceDataLoader('600900',
                                     split_point='2020-01-01',
                                     load_from_file=True),

    '601398': YahooFinanceDataLoader('601398',
                                     split_point='2020-01-01',
                                     load_from_file=True)
}


# 默认参数
def get_default_param():
    return SimpleNamespace(
        FEATURE_SIZE=64,  # 又名 n_classes
        WINDOW_SIZE=10,
        TRANSACTION_COST=0,
        BATCH_SIZE=10,
        GAMMA=0.7,
        REPLAY_MEMORY_SIZE=20,
        TARGET_UPDATE=5,
        N_STEP=8,
        N_EPISODES=1,
        HIDDEN_SIZE=64,
        WEIGHT_LIST=[
            [(9, 1, 1, 1), 10],
            [(1, 9, 1, 1), 10],
            [(1, 1, 9, 1), 10],
            [(1, 1, 1, 9), 10],
            [(8, 5, 5, 5), 30],
            [(5, 5, 8, 5), 30],
            [(8, 5, 8, 5), 30],
            [(9, 4, 7, 4), 30],
        ],
        # ALPHA_EXTENSION=[f"alpha_{('000' + str(i))[-3:]}" for i in range(1, 102)],
        # ALPHA_EXTENSION=['alpha_019', 'alpha_026', 'alpha_024', 'alpha_074', 'alpha_081',
        #                  'alpha_032', 'alpha_050', 'alpha_099', 'alpha_088', 'alpha_061',],
        # ALPHA_EXTENSION=['alpha_032', 'alpha_019', 'alpha_050', 'alpha_026', 'alpha_024',   # 互信息
        #                  'alpha_088', 'alpha_061', 'alpha_074', 'alpha_081', 'alpha_060',
        #                  'alpha_095', 'alpha_099', 'alpha_077', 'alpha_071', 'alpha_064',
        #                  'alpha_068', 'alpha_031', 'alpha_075', 'alpha_052', 'alpha_086',
        #                  'alpha_094', 'alpha_037', 'alpha_013', 'alpha_023', 'alpha_025',
        #                  'alpha_028', 'alpha_065', 'alpha_016', 'alpha_042', 'alpha_040',
        #                  'alpha_003', 'alpha_073', 'alpha_083', ],
        # ALPHA_EXTENSION=[   # spearman 相关系数
        #     'alpha_057',
        #     'alpha_083',
        #     'alpha_024',
        #     'alpha_084',
        #     'alpha_019',
        #     'alpha_013',
        #     'alpha_054',
        #     'alpha_042',
        #     'alpha_095',
        #     'alpha_002',
        #     'alpha_088',
        #     'alpha_052',
        #     'alpha_099',
        #     'alpha_077',
        #     'alpha_040',
        #     'alpha_094',
        #     'alpha_016',
        #     'alpha_003',
        #     'alpha_014',
        #     'alpha_026',
        #     'alpha_041',
        #     'alpha_046',
        #     'alpha_071',
        #     'alpha_043',
        #     'alpha_051',
        #     'alpha_017',
        #     'alpha_011',
        #     'alpha_032',
        #     'alpha_044',
        #     'alpha_053',
        # ],
        # ALPHA_EXTENSION=[   # Pearson 相关系数
        #     'alpha_032',
        #     'alpha_024',
        #     'alpha_088',
        #     'alpha_014',
        #     'alpha_077',
        #     'alpha_005',
        #     'alpha_040',
        #     'alpha_006',
        #     'alpha_004',
        #     'alpha_078',
        #     'alpha_026',
        #     'alpha_092',
        #     'alpha_094',
        #     'alpha_099',
        #     'alpha_018',
        #     'alpha_060',
        #     'alpha_047',
        #     'alpha_028',
        #     'alpha_096',
        #     'alpha_021',
        #     'alpha_003',
        #     'alpha_071',
        #     'alpha_007',
        #     'alpha_036',
        #     'alpha_054',
        #     'alpha_037',
        #     'alpha_050',
        #     'alpha_019',
        #     'alpha_035',
        #     'alpha_038',
        # ],
        ALPHA_EXTENSION=[  # 随机森林分类
            'alpha_032', 'alpha_028', 'alpha_031', 'alpha_085', 'alpha_022',
            'alpha_052', 'alpha_077', 'alpha_039', 'alpha_011', 'alpha_060',
            'alpha_066', 'alpha_030', 'alpha_040', 'alpha_072', 'alpha_055',
            'alpha_002', 'alpha_019', 'alpha_036', 'alpha_014', 'alpha_006',
            'alpha_037', 'alpha_084', 'alpha_024', 'alpha_001', 'alpha_044',
            'alpha_016', 'alpha_005', 'alpha_029', 'alpha_026', 'alpha_008',
        ],
    )


DATA_LIST = [
    '000651',
    # '000725',
    # '000858',
    # '600030',
    # '600036',
    # '600276',
    # '600519',
    # '600887',
    # '600900',
    # '601398',
]


class SensitivityRun:
    def __init__(self,
                 dataset_name,
                 gamma,
                 batch_size,
                 replay_memory_size,
                 feature_size,
                 target_update,
                 n_episodes,
                 n_step,
                 window_size,
                 device,
                 evaluation_parameter='gamma',
                 transaction_cost=0,
                 alpha_extension=None):
        """

        @param data_loader:
        @param dataset_name:
        @param gamma:
        @param batch_size:
        @param replay_memory_size:
        @param feature_size:
        @param target_update:
        @param n_episodes:
        @param n_step:
        @param window_size:
        @param device:
        @param evaluation_parameter: shows which parameter are we evaluating and can be: 'gamma', 'batch size',
            or 'replay memory size'
        @param transaction_cost:
        """
        self.data_loader = DATA_LOADERS[dataset_name]
        self.dataset_name = dataset_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.feature_size = feature_size
        self.target_update = target_update
        self.n_episodes = n_episodes
        self.n_step = n_step
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.device = device
        self.evaluation_parameter = evaluation_parameter
        self.alpha_extension = alpha_extension
        # The state mode is only for autoPatternExtractionAgent. Therefore, for pattern inputs, the state mode would be
        # set to None, because it can be recovered from the name of the data loader (e.g. dataTrain_patternBased).

        self.STATE_MODE_OHLC = 1
        self.STATE_MODE_CANDLE_REP = 4  # %body + %upper-shadow + %lower-shadow
        self.STATE_MODE_WINDOWED = 5  # window with k candles inside + the trend of those candles

        self.dataTrain_autoPatternExtractionAgent = None
        self.dataTest_autoPatternExtractionAgent = None
        self.dataTrain_patternBased = None
        self.dataTest_patternBased = None
        self.dataTrain_autoPatternExtractionAgent_candle_rep = None
        self.dataTest_autoPatternExtractionAgent_candle_rep = None
        self.dataTrain_autoPatternExtractionAgent_windowed = None
        self.dataTrain_autoPatternExtractionAgent_windowed_ext = None
        self.dataTest_autoPatternExtractionAgent_windowed = None
        self.dataTest_autoPatternExtractionAgent_windowed_ext = None
        self.dataTrain_sequential = None
        self.dataTrain_sequential_ext = None
        self.dataTest_sequential = None
        self.dataTest_sequential_ext = None
        self.dqn_pattern = None
        self.dqn_vanilla = None
        self.dqn_candle_rep = None
        self.dqn_windowed = None
        self.dqn_windowed_ext = None
        self.mlp_pattern = None
        self.mlp_vanilla = None
        self.mlp_candle_rep = None
        self.mlp_windowed = None
        self.mlp_windowed_ext = None
        self.cnn1d = None
        self.cnn2d = None
        self.cnn2d_ext = None
        self.gru = None
        self.gru_ext = None
        self.deep_cnn = None
        self.cnn_gru = None
        self.cnn_gru_ext = None
        self.cnn_attn = None
        self.experiment_path = os.path.join(os.path.abspath(os.getcwd()),
                                            'Results/' + self.evaluation_parameter + '/')
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

        self.reset()
        self.test_portfolios = {
            # 'DQN-pattern': {},
            # 'DQN-vanilla': {},
            # 'DQN-candlerep': {},
            'DQN-windowed': {},
            'DQN-windowed-ext': {},
            # 'MLP-pattern': {},
            # 'MLP-vanilla': {},
            # 'MLP-candlerep': {},
            'MLP-windowed': {},
            'MLP-windowed-ext': {},
            # 'CNN1d': {},
            'CNN2d': {},
            'CNN2d-ext': {},
            'GRU': {},
            'GRU-ext': {},
            # 'Deep-CNN': {},
            'CNN-GRU': {},
            'CNN-GRU-ext': {},
            # 'CNN-ATTN': {},
            'BAH': {}
        }

    def reset(self):
        self.load_data()
        self.load_agents()

    def load_data(self):
        self.dataTrain_autoPatternExtractionAgent = \
            DataAutoPatternExtractionAgent(self.data_loader.data_train,
                                           self.STATE_MODE_OHLC,
                                           'action_auto_pattern_extraction',
                                           self.device,
                                           self.gamma,
                                           self.n_step,
                                           self.batch_size,
                                           self.window_size,
                                           self.transaction_cost)

        self.dataTest_autoPatternExtractionAgent = \
            DataAutoPatternExtractionAgent(self.data_loader.data_test,
                                           self.STATE_MODE_OHLC,
                                           'action_auto_pattern_extraction',
                                           self.device,
                                           self.gamma,
                                           self.n_step,
                                           self.batch_size,
                                           self.window_size,
                                           self.transaction_cost)

        self.dataTrain_patternBased = \
            DataForPatternBasedAgent(self.data_loader.data_train,
                                     self.data_loader.patterns,
                                     'action_pattern',
                                     self.device, self.gamma,
                                     self.n_step, self.batch_size,
                                     self.transaction_cost)

        self.dataTest_patternBased = \
            DataForPatternBasedAgent(self.data_loader.data_test,
                                     self.data_loader.patterns,
                                     'action_pattern',
                                     self.device,
                                     self.gamma,
                                     self.n_step,
                                     self.batch_size,
                                     self.transaction_cost)

        self.dataTrain_autoPatternExtractionAgent_candle_rep = \
            DataAutoPatternExtractionAgent(
                self.data_loader.data_train,
                self.STATE_MODE_CANDLE_REP,
                'action_candle_rep',
                self.device,
                self.gamma, self.n_step, self.batch_size,
                self.window_size,
                self.transaction_cost)
        self.dataTest_autoPatternExtractionAgent_candle_rep = \
            DataAutoPatternExtractionAgent(self.data_loader.data_test,
                                           self.STATE_MODE_CANDLE_REP,
                                           'action_candle_rep',
                                           self.device,
                                           self.gamma, self.n_step,
                                           self.batch_size,
                                           self.window_size,
                                           self.transaction_cost)

        self.dataTrain_autoPatternExtractionAgent_windowed = \
            DataAutoPatternExtractionAgent(self.data_loader.data_train,
                                           self.STATE_MODE_WINDOWED,
                                           'action_auto_extraction_windowed',
                                           self.device,
                                           self.gamma, self.n_step,
                                           self.batch_size,
                                           self.window_size,
                                           self.transaction_cost)
        self.dataTest_autoPatternExtractionAgent_windowed = \
            DataAutoPatternExtractionAgent(self.data_loader.data_test,
                                           self.STATE_MODE_WINDOWED,
                                           'action_auto_extraction_windowed',
                                           self.device,
                                           self.gamma, self.n_step,
                                           self.batch_size,
                                           self.window_size,
                                           self.transaction_cost)

        self.dataTrain_autoPatternExtractionAgent_windowed_ext = \
            DataAutoPatternExtractionAgent(self.data_loader.data_train,
                                           self.STATE_MODE_WINDOWED,
                                           'action_auto_extraction_windowed_ext',
                                           self.device,
                                           self.gamma, self.n_step,
                                           self.batch_size,
                                           self.window_size,
                                           self.transaction_cost,
                                           alpha_extension=self.alpha_extension)
        self.dataTest_autoPatternExtractionAgent_windowed_ext = \
            DataAutoPatternExtractionAgent(self.data_loader.data_test,
                                           self.STATE_MODE_WINDOWED,
                                           'action_auto_extraction_windowed_ext',
                                           self.device,
                                           self.gamma, self.n_step,
                                           self.batch_size,
                                           self.window_size,
                                           self.transaction_cost,
                                           alpha_extension=self.alpha_extension)

        self.dataTrain_sequential = DataSequential(self.data_loader.data_train,
                                                   'action_sequential',
                                                   self.device,
                                                   self.gamma,
                                                   self.n_step,
                                                   self.batch_size,
                                                   self.window_size,
                                                   self.transaction_cost)

        self.dataTest_sequential = DataSequential(self.data_loader.data_test,
                                                  'action_sequential',
                                                  self.device,
                                                  self.gamma,
                                                  self.n_step,
                                                  self.batch_size,
                                                  self.window_size,
                                                  self.transaction_cost)

        self.dataTrain_sequential_ext = DataSequential(self.data_loader.data_train,
                                                       'action_sequential',
                                                       self.device,
                                                       self.gamma,
                                                       self.n_step,
                                                       self.batch_size,
                                                       self.window_size,
                                                       self.transaction_cost,
                                                       alpha_extension=self.alpha_extension)

        self.dataTest_sequential_ext = DataSequential(self.data_loader.data_test,
                                                      'action_sequential',
                                                      self.device,
                                                      self.gamma,
                                                      self.n_step,
                                                      self.batch_size,
                                                      self.window_size,
                                                      self.transaction_cost,
                                                      alpha_extension=self.alpha_extension)

    def load_agents(self):
        self.dqn_pattern = DeepRL(self.data_loader,
                                  self.dataTrain_patternBased,
                                  self.dataTest_patternBased,
                                  self.dataset_name,
                                  None,
                                  self.window_size,
                                  self.transaction_cost,
                                  BATCH_SIZE=self.batch_size,
                                  GAMMA=self.gamma,
                                  ReplayMemorySize=self.replay_memory_size,
                                  TARGET_UPDATE=self.target_update,
                                  n_step=self.n_step)

        self.dqn_vanilla = DeepRL(self.data_loader,
                                  self.dataTrain_autoPatternExtractionAgent,
                                  self.dataTest_autoPatternExtractionAgent,
                                  self.dataset_name,
                                  self.STATE_MODE_OHLC,
                                  self.window_size,
                                  self.transaction_cost,
                                  BATCH_SIZE=self.batch_size,
                                  GAMMA=self.gamma,
                                  ReplayMemorySize=self.replay_memory_size,
                                  TARGET_UPDATE=self.target_update,
                                  n_step=self.n_step)

        self.dqn_candle_rep = DeepRL(self.data_loader,
                                     self.dataTrain_autoPatternExtractionAgent_candle_rep,
                                     self.dataTest_autoPatternExtractionAgent_candle_rep,
                                     self.dataset_name,
                                     self.STATE_MODE_CANDLE_REP,
                                     self.window_size,
                                     self.transaction_cost,
                                     BATCH_SIZE=self.batch_size,
                                     GAMMA=self.gamma,
                                     ReplayMemorySize=self.replay_memory_size,
                                     TARGET_UPDATE=self.target_update,
                                     n_step=self.n_step)

        self.dqn_windowed = DeepRL(self.data_loader,
                                   self.dataTrain_autoPatternExtractionAgent_windowed,
                                   self.dataTest_autoPatternExtractionAgent_windowed,
                                   self.dataset_name,
                                   self.STATE_MODE_WINDOWED,
                                   self.window_size,
                                   self.transaction_cost,
                                   BATCH_SIZE=self.batch_size,
                                   GAMMA=self.gamma,
                                   ReplayMemorySize=self.replay_memory_size,
                                   TARGET_UPDATE=self.target_update,
                                   n_step=self.n_step)

        self.dqn_windowed_ext = DeepRL(self.data_loader,
                                       self.dataTrain_autoPatternExtractionAgent_windowed_ext,
                                       self.dataTest_autoPatternExtractionAgent_windowed_ext,
                                       self.dataset_name,
                                       self.STATE_MODE_WINDOWED,
                                       self.window_size,
                                       self.transaction_cost,
                                       BATCH_SIZE=self.batch_size,
                                       GAMMA=self.gamma,
                                       ReplayMemorySize=self.replay_memory_size,
                                       TARGET_UPDATE=self.target_update,
                                       n_step=self.n_step)

        self.mlp_pattern = SimpleMLP(self.data_loader,
                                     self.dataTrain_patternBased,
                                     self.dataTest_patternBased,
                                     self.dataset_name,
                                     None,
                                     self.window_size,
                                     self.transaction_cost,
                                     self.feature_size,
                                     BATCH_SIZE=self.batch_size,
                                     GAMMA=self.gamma,
                                     ReplayMemorySize=self.replay_memory_size,
                                     TARGET_UPDATE=self.target_update,
                                     n_step=self.n_step)

        self.mlp_vanilla = SimpleMLP(self.data_loader,
                                     self.dataTrain_autoPatternExtractionAgent,
                                     self.dataTest_autoPatternExtractionAgent,
                                     self.dataset_name,
                                     self.STATE_MODE_OHLC,
                                     self.window_size,
                                     self.transaction_cost,
                                     self.feature_size,
                                     BATCH_SIZE=self.batch_size,
                                     GAMMA=self.gamma,
                                     ReplayMemorySize=self.replay_memory_size,
                                     TARGET_UPDATE=self.target_update,
                                     n_step=self.n_step)

        self.mlp_candle_rep = SimpleMLP(self.data_loader,
                                        self.dataTrain_autoPatternExtractionAgent_candle_rep,
                                        self.dataTest_autoPatternExtractionAgent_candle_rep,
                                        self.dataset_name,
                                        self.STATE_MODE_CANDLE_REP,
                                        self.window_size,
                                        self.transaction_cost,
                                        self.feature_size,
                                        BATCH_SIZE=self.batch_size,
                                        GAMMA=self.gamma,
                                        ReplayMemorySize=self.replay_memory_size,
                                        TARGET_UPDATE=self.target_update,
                                        n_step=self.n_step)

        self.mlp_windowed = SimpleMLP(self.data_loader,
                                      self.dataTrain_autoPatternExtractionAgent_windowed,
                                      self.dataTest_autoPatternExtractionAgent_windowed,
                                      self.dataset_name,
                                      self.STATE_MODE_WINDOWED,
                                      self.window_size,
                                      self.transaction_cost,
                                      self.feature_size,
                                      BATCH_SIZE=self.batch_size,
                                      GAMMA=self.gamma,
                                      ReplayMemorySize=self.replay_memory_size,
                                      TARGET_UPDATE=self.target_update,
                                      n_step=self.n_step)

        self.mlp_windowed_ext = SimpleMLP(self.data_loader,
                                          self.dataTrain_autoPatternExtractionAgent_windowed_ext,
                                          self.dataTest_autoPatternExtractionAgent_windowed_ext,
                                          self.dataset_name,
                                          self.STATE_MODE_WINDOWED,
                                          self.window_size,
                                          self.transaction_cost,
                                          self.feature_size,
                                          BATCH_SIZE=self.batch_size,
                                          GAMMA=self.gamma,
                                          ReplayMemorySize=self.replay_memory_size,
                                          TARGET_UPDATE=self.target_update,
                                          n_step=self.n_step)

        self.cnn1d = SimpleCNN(self.data_loader,
                               self.dataTrain_autoPatternExtractionAgent,
                               self.dataTest_autoPatternExtractionAgent,
                               self.dataset_name,
                               self.STATE_MODE_OHLC,
                               self.window_size,
                               self.transaction_cost,
                               self.feature_size,
                               BATCH_SIZE=self.batch_size,
                               GAMMA=self.gamma,
                               ReplayMemorySize=self.replay_memory_size,
                               TARGET_UPDATE=self.target_update,
                               n_step=self.n_step)

        self.cnn2d = CNN2d(self.data_loader,
                           self.dataTrain_sequential,
                           self.dataTest_sequential,
                           self.dataset_name,
                           self.feature_size,
                           self.transaction_cost,
                           BATCH_SIZE=self.batch_size,
                           GAMMA=self.gamma,
                           ReplayMemorySize=self.replay_memory_size,
                           TARGET_UPDATE=self.target_update,
                           n_step=self.n_step,
                           window_size=self.window_size)

        self.cnn2d_ext = CNN2d(self.data_loader,
                               self.dataTrain_sequential_ext,
                               self.dataTest_sequential_ext,
                               self.dataset_name,
                               self.feature_size,
                               self.transaction_cost,
                               BATCH_SIZE=self.batch_size,
                               GAMMA=self.gamma,
                               ReplayMemorySize=self.replay_memory_size,
                               TARGET_UPDATE=self.target_update,
                               n_step=self.n_step,
                               window_size=self.window_size)

        self.gru = GRU(self.data_loader,
                       self.dataTrain_sequential,
                       self.dataTest_sequential,
                       self.dataset_name,
                       self.transaction_cost,
                       self.feature_size,
                       BATCH_SIZE=self.batch_size,
                       GAMMA=self.gamma,
                       ReplayMemorySize=self.replay_memory_size,
                       TARGET_UPDATE=self.target_update,
                       n_step=self.n_step,
                       window_size=self.window_size)

        self.gru_ext = GRU(self.data_loader,
                           self.dataTrain_sequential_ext,
                           self.dataTest_sequential_ext,
                           self.dataset_name,
                           self.transaction_cost,
                           self.feature_size,
                           BATCH_SIZE=self.batch_size,
                           GAMMA=self.gamma,
                           ReplayMemorySize=self.replay_memory_size,
                           TARGET_UPDATE=self.target_update,
                           n_step=self.n_step,
                           window_size=self.window_size)

        self.deep_cnn = CNN(self.data_loader,
                            self.dataTrain_sequential,
                            self.dataTest_sequential,
                            self.dataset_name,
                            self.transaction_cost,
                            BATCH_SIZE=self.batch_size,
                            GAMMA=self.gamma,
                            ReplayMemorySize=self.replay_memory_size,
                            TARGET_UPDATE=self.target_update,
                            n_step=self.n_step,
                            window_size=self.window_size)

        self.cnn_gru = CNN_GRU(self.data_loader,
                               self.dataTrain_sequential,
                               self.dataTest_sequential,
                               self.dataset_name,
                               self.transaction_cost,
                               self.feature_size,
                               BATCH_SIZE=self.batch_size,
                               GAMMA=self.gamma,
                               ReplayMemorySize=self.replay_memory_size,
                               TARGET_UPDATE=self.target_update,
                               n_step=self.n_step,
                               window_size=self.window_size)

        self.cnn_gru_ext = CNN_GRU(self.data_loader,
                                   self.dataTrain_sequential_ext,
                                   self.dataTest_sequential_ext,
                                   self.dataset_name,
                                   self.transaction_cost,
                                   self.feature_size,
                                   BATCH_SIZE=self.batch_size,
                                   GAMMA=self.gamma,
                                   ReplayMemorySize=self.replay_memory_size,
                                   TARGET_UPDATE=self.target_update,
                                   n_step=self.n_step,
                                   window_size=self.window_size)

        self.cnn_attn = CNN_ATTN(self.data_loader,
                                 self.dataTrain_sequential,
                                 self.dataTest_sequential,
                                 self.dataset_name,
                                 self.transaction_cost,
                                 self.feature_size,
                                 BATCH_SIZE=self.batch_size,
                                 GAMMA=self.gamma,
                                 ReplayMemorySize=self.replay_memory_size,
                                 TARGET_UPDATE=self.target_update,
                                 n_step=self.n_step,
                                 window_size=self.window_size)

    def train(self):
        # self.dqn_pattern.train(self.n_episodes)
        # self.dqn_vanilla.train(self.n_episodes)
        # self.dqn_candle_rep.train(self.n_episodes)
        self.dqn_windowed.train(self.n_episodes)
        self.dqn_windowed_ext.train(self.n_episodes)
        # self.mlp_pattern.train(self.n_episodes)
        # self.mlp_vanilla.train(self.n_episodes)
        # self.mlp_candle_rep.train(self.n_episodes)
        self.mlp_windowed.train(self.n_episodes)
        self.mlp_windowed_ext.train(self.n_episodes)
        # self.cnn1d.train(self.n_episodes)
        self.cnn2d.train(self.n_episodes)
        self.cnn2d_ext.train(self.n_episodes)
        self.gru.train(self.n_episodes)
        self.gru_ext.train(self.n_episodes)
        # self.deep_cnn.train(self.n_episodes)
        self.cnn_gru.train(self.n_episodes)
        self.cnn_gru_ext.train(self.n_episodes)
        # self.cnn_attn.train(self.n_episodes)

    def evaluate_sensitivity(self):
        key = None
        if self.evaluation_parameter == 'gamma':
            key = self.gamma
        elif self.evaluation_parameter == 'batch size':
            key = self.batch_size
        elif self.evaluation_parameter == 'replay memory size':
            key = self.replay_memory_size

        # self.test_portfolios['DQN-pattern'][key] = self.dqn_pattern.test().get_daily_portfolio_value()
        # self.test_portfolios['DQN-vanilla'][key] = self.dqn_vanilla.test().get_daily_portfolio_value()
        # self.test_portfolios['DQN-candlerep'][key] = self.dqn_candle_rep.test().get_daily_portfolio_value()
        self.test_portfolios['DQN-windowed'][key] = self.dqn_windowed.test().get_daily_portfolio_value()
        self.test_portfolios['DQN-windowed-ext'][key] = self.dqn_windowed_ext.test().get_daily_portfolio_value()
        # self.test_portfolios['MLP-pattern'][key] = self.mlp_pattern.test().get_daily_portfolio_value()
        # self.test_portfolios['MLP-vanilla'][key] = self.mlp_vanilla.test().get_daily_portfolio_value()
        # self.test_portfolios['MLP-candlerep'][key] = self.mlp_candle_rep.test().get_daily_portfolio_value()
        self.test_portfolios['MLP-windowed'][key] = self.mlp_windowed.test().get_daily_portfolio_value()
        self.test_portfolios['MLP-windowed-ext'][key] = self.mlp_windowed_ext.test().get_daily_portfolio_value()
        # self.test_portfolios['CNN1d'][key] = self.cnn1d.test().get_daily_portfolio_value()
        self.test_portfolios['CNN2d'][key] = self.cnn2d.test().get_daily_portfolio_value()
        self.test_portfolios['CNN2d-ext'][key] = self.cnn2d_ext.test().get_daily_portfolio_value()
        self.test_portfolios['GRU'][key] = self.gru.test().get_daily_portfolio_value()
        self.test_portfolios['GRU-ext'][key] = self.gru_ext.test().get_daily_portfolio_value()
        # self.test_portfolios['Deep-CNN'][key] = self.deep_cnn.test().get_daily_portfolio_value()
        self.test_portfolios['CNN-GRU'][key] = self.cnn_gru.test().get_daily_portfolio_value()
        self.test_portfolios['CNN-GRU-ext'][key] = self.cnn_gru_ext.test().get_daily_portfolio_value()
        # self.test_portfolios['CNN-ATTN'][key] = self.cnn_attn.test().get_daily_portfolio_value()

    def plot_and_save_sensitivity(self):
        plot_path = os.path.join(self.experiment_path, 'plots')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        sns.set(rc={'figure.figsize': (15, 7)})
        sns.set_palette(sns.color_palette("Paired", 15))

        for model_name in self.test_portfolios.keys():
            first = True
            ax = None
            for gamma in self.test_portfolios[model_name]:
                profit_percentage = [
                    (self.test_portfolios[model_name][gamma][i] - self.test_portfolios[model_name][gamma][0]) /
                    self.test_portfolios[model_name][gamma][0] * 100
                    for i in range(len(self.test_portfolios[model_name][gamma]))]

                difference = len(self.test_portfolios[model_name][gamma]) - len(self.data_loader.data_test_with_date)
                df = pd.DataFrame({'date': self.data_loader.data_test_with_date.index,
                                   'portfolio': profit_percentage[difference:]})
                if not first:
                    df.plot(ax=ax, x='date', y='portfolio', label=gamma)
                else:
                    ax = df.plot(x='date', y='portfolio', label=gamma)
                    first = False

            if ax is not None:
                ax.set(xlabel='Time', ylabel='%Rate of Return')
                ax.set_title(f'Analyzing the sensitivity of {model_name} to {self.evaluation_parameter}')
                plt.legend()
                fig_file = os.path.join(plot_path, f'{model_name}.jpg')
                plt.savefig(fig_file, dpi=300)

    def save_portfolios(self):
        path = os.path.join(self.experiment_path, 'portfolios.pkl')
        save_pkl(path, self.test_portfolios)

    def save_experiment(self):
        self.plot_and_save_sensitivity()
        self.save_portfolios()

    def evaluate_models(self):
        self.test_portfolios['DQN-windowed'] = self.dqn_windowed.test().get_daily_portfolio_value()
        self.test_portfolios['DQN-windowed-ext'] = self.dqn_windowed_ext.test().get_daily_portfolio_value()
        self.test_portfolios['MLP-windowed'] = self.mlp_windowed.test().get_daily_portfolio_value()
        self.test_portfolios['MLP-windowed-ext'] = self.mlp_windowed_ext.test().get_daily_portfolio_value()
        self.test_portfolios['CNN2d'] = self.cnn2d.test().get_daily_portfolio_value()
        self.test_portfolios['CNN2d-ext'] = self.cnn2d_ext.test().get_daily_portfolio_value()
        self.test_portfolios['GRU'] = self.gru.test().get_daily_portfolio_value()
        self.test_portfolios['GRU-ext'] = self.gru_ext.test().get_daily_portfolio_value()
        self.test_portfolios['CNN-GRU'] = self.cnn_gru.test().get_daily_portfolio_value()
        self.test_portfolios['CNN-GRU-ext'] = self.cnn_gru_ext.test().get_daily_portfolio_value()
        self.evaluate_bah_model()

    def print_evaluate_models(self):
        check_list = [
            'dqn_windowed',
            'mlp_windowed',
            'cnn2d',
            'gru',
            'cnn_gru',
        ]

        for name in check_list:
            left_func = getattr(self, f"{name}_ext")
            right_func = getattr(self, f"{name}")
            left_val = left_func.test().total_return()
            right_val = right_func.test().total_return()
            print(
                f"=========== {(name + ' ' * 20)[:20]} Rate: {round(left_val / right_val, 2)}      {'**' if right_val <= 0 else ''}")

        # print(f"Total Return (Rate越大越好):")
        # print(f"dqn_windowed: {round(self.dqn_windowed.test().total_return(), 2)}")
        # print(f"dqn_windowed_ext: {round(self.dqn_windowed_ext.test().total_return(), 2)}")
        # print(f"=========== dqn_windowed Rate: {round(self.dqn_windowed_ext.test().total_return() / self.dqn_windowed.test().total_return(), 2)}")

        # print(f"mlp_windowed: {round(self.mlp_windowed.test().total_return(), 2)}")
        # print(f"mlp_windowed_ext: {round(self.mlp_windowed_ext.test().total_return(), 2)}")
        # print(f"=========== mlp_windowed Rate: {round(self.mlp_windowed_ext.test().total_return() / self.mlp_windowed.test().total_return(), 2)}")

        # print(f"cnn2d: {round(self.cnn2d.test().total_return(), 2)}")
        # print(f"cnn2d_ext: {round(self.cnn2d_ext.test().total_return(), 2)}")
        # print(f"=========== cnn2d Rate:        {round(self.cnn2d_ext.test().total_return() / self.cnn2d.test().total_return(), 2)}")

        # print(f"gru: {round(self.gru.test().total_return(), 2)}")
        # print(f"gru_ext: {round(self.gru_ext.test().total_return(), 2)}")
        # print(f"=========== gru Rate:          {round(self.gru_ext.test().total_return() / self.gru.test().total_return(), 2)}")

        # print(f"cnn_gru: {round(self.cnn_gru.test().total_return(), 2)}")
        # print(f"cnn_gru_ext: {round(self.cnn_gru_ext.test().total_return(), 2)}")
        # print(f"=========== cnn_gru Rate:      {round(self.cnn_gru_ext.test().total_return() / self.cnn_gru.test().total_return(), 2)}")

    def evaluate_bah_model(self):
        self.dataTest_patternBased.data[self.dataTest_patternBased.action_name] = 'buy'
        ev_bah = Evaluation(self.dataTest_patternBased.data, self.dataTest_patternBased.action_name, 1000)
        self.test_portfolios['BAH'] = ev_bah.get_daily_portfolio_value()

    def save_models_compare(self):
        self.plot_and_save_models_compare()
        self.save_portfolios()
        pass

    def plot_and_save_models_compare(self):
        plot_path = os.path.join(self.experiment_path, 'plots')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        sns.set(rc={'figure.figsize': (15, 7)})
        sns.set_palette(sns.color_palette("Paired", 15))

        first = True
        ax = None
        for model_name in self.test_portfolios.keys():
            profit_percentage = [
                (self.test_portfolios[model_name][i] - self.test_portfolios[model_name][0]) /
                self.test_portfolios[model_name][0] * 100
                for i in range(len(self.test_portfolios[model_name]))]
            difference = len(self.test_portfolios[model_name]) - len(self.data_loader.data_test_with_date)
            df = pd.DataFrame({'date': pd.to_datetime(self.data_loader.data_test_with_date.index),
                               'portfolio': profit_percentage[difference:]})
            if not first:
                df.plot(ax=ax, x='date', y='portfolio', label=model_name)
            else:
                ax = df.plot(x='date', y='portfolio', label=model_name)
                first = False

            # 在每条线的末尾添加标签
            last_date = df['date'].iloc[-1]  # 获取最后一个日期
            last_value = df['portfolio'].iloc[-1]  # 获取最后一个值
            offset = pd.Timedelta(days=5)
            new_date = last_date + offset  # 向右移动的新位置
            # 添加标签（确保new_date已经是datetime类型）
            ax.text(new_date, last_value, model_name, horizontalalignment='left')

        if ax is not None:
            ax.set(xlabel='Time', ylabel='%Rate of Return')
            ax.set_title(f'Compare Models Return For {self.dataset_name}')
            plt.legend()
            fig_file = os.path.join(plot_path, f'{self.dataset_name}.jpg')
            plt.savefig(fig_file, dpi=300)

        pass


def plot_portfolios(portfolios_data, data_test_with_date, file_path, file_name, title="Total Return"):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    sns.set(rc={'figure.figsize': (15, 7)})
    sns.set_palette(sns.color_palette("Paired", 15))

    first = True
    ax = None
    for key in portfolios_data.keys():
        profit_percentage = [
            (portfolios_data[key][i] - portfolios_data[key][0]) / portfolios_data[key][0] * 100
            for i in range(len(portfolios_data[key]))
        ]

        difference = len(portfolios_data[key]) - len(data_test_with_date)
        df = pd.DataFrame({'date': data_test_with_date.index,
                           'portfolio': profit_percentage[difference:]})
        if not first:
            df.plot(ax=ax, x='date', y='portfolio', label=key)
        else:
            ax = df.plot(x='date', y='portfolio', label=key)
            first = False

    if ax is not None:
        ax.set(xlabel='Time', ylabel='%Rate of Return')
        ax.set_title(title)
        plt.legend()
        fig_file = os.path.join(file_path, f'{file_name}.jpg')
        plt.savefig(fig_file, dpi=300)


def save_portfolios(portfolios_data, file_path, file_name):
    path = os.path.join(file_path, f"{file_name}_portfolios.pkl")
    save_pkl(path, portfolios_data)


def train_mlp_windowed_ext(data_name, data_loader, portfolios_data, label_name=f"MLP_windowed_ext", param=None,
                           weight=(1, 1, 1, 1), epoches=10):
    if param is None:
        param = get_default_param()
    trainData = DataAutoPatternExtractionAgent(data_loader.data_train,
                                               5,
                                               'action_auto_extraction_windowed_and_ext',
                                               device, param.GAMMA, param.N_STEP, param.BATCH_SIZE,
                                               param.WINDOW_SIZE, param.TRANSACTION_COST, param.ALPHA_EXTENSION)
    testData = DataAutoPatternExtractionAgent(data_loader.data_test,
                                              5,
                                              'action_auto_extraction_windowed_and_ext',
                                              device, param.GAMMA, param.N_STEP, param.BATCH_SIZE,
                                              param.WINDOW_SIZE, param.TRANSACTION_COST, param.ALPHA_EXTENSION)
    mlp_windowed_ext = SimpleMLPExt(data_loader, trainData, testData, data_name,
                                    state_mode=5, window_size=param.WINDOW_SIZE,
                                    hidden_size=param.HIDDEN_SIZE, weight=weight,
                                    transaction_cost=param.TRANSACTION_COST,
                                    n_classes=param.FEATURE_SIZE,
                                    BATCH_SIZE=param.BATCH_SIZE, GAMMA=param.GAMMA,
                                    ReplayMemorySize=param.REPLAY_MEMORY_SIZE,
                                    TARGET_UPDATE=param.TARGET_UPDATE, n_step=param.N_STEP)
    print(f"================ Encoder: \n{mlp_windowed_ext.encoder}")
    print(f"================ Decoder: \n{mlp_windowed_ext.policy_decoder}")
    print(f"================ weight : \n{weight}")
    mlp_windowed_ext.train(num_episodes=epoches, test_per_epoch=True,
                           tag=f"{data_loader.DATA_NAME}-({'-'.join([str(i) for i in weight])})")
    mlp_windowed_ext.test().evaluate(simple_print=True)
    portfolios_data[label_name] = mlp_windowed_ext.test().get_daily_portfolio_value()


def train_gru(data_name, data_loader, portfolios_data, label_name=f"GRU", param=None):
    if param is None:
        param = get_default_param()
    trainData = DataSequential(data_loader.data_train,
                               'action_sequential',
                               device, param.GAMMA, param.N_STEP, param.BATCH_SIZE,
                               param.WINDOW_SIZE, param.TRANSACTION_COST)
    testData = DataSequential(data_loader.data_test,
                              'action_sequential',
                              device, param.GAMMA, param.N_STEP, param.BATCH_SIZE,
                              param.WINDOW_SIZE, param.TRANSACTION_COST)
    gru = GRU(data_loader, trainData, testData, data_name, param.TRANSACTION_COST, param.FEATURE_SIZE,
              BATCH_SIZE=param.BATCH_SIZE, GAMMA=param.GAMMA, ReplayMemorySize=param.REPLAY_MEMORY_SIZE,
              TARGET_UPDATE=param.TARGET_UPDATE, n_step=param.N_STEP, window_size=param.WINDOW_SIZE)
    print(f"================ Encoder: \n{gru.encoder}")
    print(f"================ Decoder: \n{gru.policy_decoder}")
    gru.train(num_episodes=param.N_EPISODES, tensorboard=tensorboard_writer)
    gru.test().evaluate()
    portfolios_data[label_name] = gru.test().get_daily_portfolio_value()


def train_gru_ext(data_name, data_loader, portfolios_data, label_name=f"GRU_ext", param=None):
    if param is None:
        param = get_default_param()
    trainData = DataSequential(data_loader.data_train,
                               'action_sequential',
                               device, param.GAMMA, param.N_STEP, param.BATCH_SIZE,
                               param.WINDOW_SIZE, param.TRANSACTION_COST, param.ALPHA_EXTENSION)
    testData = DataSequential(data_loader.data_test,
                              'action_sequential',
                              device, param.GAMMA, param.N_STEP, param.BATCH_SIZE,
                              param.WINDOW_SIZE, param.TRANSACTION_COST, param.ALPHA_EXTENSION)
    gru_ext = GRUExt(data_loader, trainData, testData, data_name, param.TRANSACTION_COST, param.FEATURE_SIZE,
                     BATCH_SIZE=param.BATCH_SIZE, GAMMA=param.GAMMA, ReplayMemorySize=param.REPLAY_MEMORY_SIZE,
                     TARGET_UPDATE=param.TARGET_UPDATE, n_step=param.N_STEP, window_size=param.WINDOW_SIZE)
    print(f"================ Encoder: \n{gru_ext.encoder}")
    print(f"================ Decoder: \n{gru_ext.policy_decoder}")
    gru_ext.train(num_episodes=param.N_EPISODES, tensorboard=tensorboard_writer)
    gru_ext.test().evaluate()
    portfolios_data[label_name] = gru_ext.test().get_daily_portfolio_value()


def train_mlp_windowed(data_name, data_loader, portfolios_data, label_name=f"MLP_windowed", param=None,
                       load_pre_model=False):
    if param is None:
        param = get_default_param()
    trainData = DataAutoPatternExtractionAgent(data_loader.data_train,
                                               5,
                                               'action_auto_extraction_windowed',
                                               device, param.GAMMA, param.N_STEP, param.BATCH_SIZE,
                                               param.WINDOW_SIZE, param.TRANSACTION_COST)
    testData = DataAutoPatternExtractionAgent(data_loader.data_test,
                                              5,
                                              'action_auto_extraction_windowed',
                                              device, param.GAMMA, param.N_STEP, param.BATCH_SIZE,
                                              param.WINDOW_SIZE, param.TRANSACTION_COST)
    mlp_windowed = SimpleMLP(data_loader, trainData, testData, data_name,
                             state_mode=5, window_size=param.WINDOW_SIZE, transaction_cost=param.TRANSACTION_COST,
                             n_classes=param.FEATURE_SIZE,
                             BATCH_SIZE=param.BATCH_SIZE, GAMMA=param.GAMMA, ReplayMemorySize=param.REPLAY_MEMORY_SIZE,
                             TARGET_UPDATE=param.TARGET_UPDATE, n_step=param.N_STEP)
    print(f"================ Encoder: \n{mlp_windowed.encoder}")
    print(f"================ Decoder: \n{mlp_windowed.policy_decoder}")
    if load_pre_model:
        mlp_windowed.load_model()
        mlp_windowed.test().evaluate(simple_print=True)
    mlp_windowed.train(num_episodes=param.N_EPISODES)
    mlp_windowed.test().evaluate(simple_print=True)
    portfolios_data[label_name] = mlp_windowed.test().get_daily_portfolio_value()
    return mlp_windowed


def plot_actions(agent, data_loader, file_path, file_name):
    df1 = data_loader.data_test_with_date
    action_list = list(agent.data_test.data[agent.data_test.action_name])
    df1[agent.data_test.action_name] = action_list

    df1 = df1

    # 准备买入、卖出和持有的数据
    buy = df1[df1[agent.data_test.action_name] == 'buy']['close']
    sell = df1[df1[agent.data_test.action_name] == 'sell']['close']
    none = df1[df1[agent.data_test.action_name] == 'None']['close']

    # 创建价格线（收盘价）
    price_line = go.Scatter(x=df1.index, y=df1['close'], mode='lines', name='收盘价', line=dict(color='blue'))

    # 添加买入标记（红色向上三角形）
    buy_marker = go.Scatter(x=buy.index, y=buy, mode='markers', name='买入',
                            marker=dict(symbol='triangle-up', color='red', size=10))

    # 添加卖出标记（绿色向下三角形）
    sell_marker = go.Scatter(x=sell.index, y=sell, mode='markers', name='卖出',
                             marker=dict(symbol='triangle-down', color='green', size=10))

    # 添加持有标记（灰色圆点）
    none_marker = go.Scatter(x=none.index, y=none, mode='markers', name='持有',
                             marker=dict(symbol='circle', color='grey', size=10))

    # 定义图表布局
    layout = go.Layout(
        autosize=False,
        width=900,
        height=600,
        xaxis=dict(
            rangeslider=dict(visible=False),
        )
    )

    # 创建图表对象并添加数据
    figSignal = go.Figure(data=[price_line, buy_marker, sell_marker, none_marker], layout=layout)

    # 确保文件路径存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 保存图表为图片
    figSignal.write_image(os.path.join(file_path, f"{file_name}.jpg"), format='jpeg')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch version: {torch.__version__}")
    print(f"cal device: {device}")

    start_time = dt.datetime.now()
    print(f"--------- total start time: {start_time}")

    for data_name in ['BTC-USD']:
        data_loader = DATA_LOADERS[data_name]
        portfolios_data = {}
        param = get_default_param()
        agent = train_mlp_windowed(data_name,
                                   data_loader,
                                   portfolios_data,
                                   label_name="MLP_windowed",
                                   param=param,
                                   load_pre_model=True)

        file_path = "Results/V2Test"
        plot_portfolios(portfolios_data, data_loader.data_test_with_date, file_path, f"{data_name}-V2Test-Return")
        plot_actions(agent, data_loader, file_path, f"{data_name}-Vf2Test-Action")
        save_portfolios(portfolios_data, file_path, f"{data_name}-V2Test")

    # 回报结果

    # for data_name in DATA_LIST:
    #     data_loader = DATA_LOADERS[data_name]
    #     portfolios_data = {}
    #
    #     param = get_default_param()
    #     for [weight, eps] in param.WEIGHT_LIST:
    #         train_mlp_windowed_ext(data_name, data_loader, portfolios_data,
    #                                label_name=f"FullExt_({'-'.join([str(item) for item in weight])})", param=param,
    #                                weight=weight, epoches=eps)
    #
    #     file_path = "Results/ModelCompare"
    #     plot_portfolios(portfolios_data, data_loader.data_test_with_date, file_path, f"{data_name}-MyModel-Big")
    #     save_portfolios(portfolios_data, file_path, f"{data_name}-MyModel-Big")

    # param = get_default_param()
    # train_gru(data_name, data_loader, portfolios_data, label_name=f"GRU-epc5", param=param)

    # param = get_default_param()
    # train_gru_ext(data_name, data_loader, portfolios_data, label_name=f"GRUExt-epc5", param=param)

    # '''
    # ---------- 以下开始分模型获取数据
    # '''
    # # 以下开始分模型获取数据
    # feature_size_list = [16, 64, 256]
    # window_size_list = [10, 20, 40]
    # gamma_list = [0.7, 0.8, 0.9]
    # n_step_list = [3, 8, 20]
    #
    # # feature_size
    # portfolios_data = {}
    # param = get_default_param()
    # for feature_size in feature_size_list:
    #     print(f"{'='*50} compare feature_size {feature_size}")
    #     param.FEATURE_SIZE = feature_size
    #     train_mlp_windowed(data_name, data_loader, portfolios_data, label_name=f"MLP-feature{feature_size}", param=param)
    #     train_mlp_windowed_ext(data_name, data_loader, portfolios_data, label_name=f"MLPExt-feature{feature_size}", param=param)
    #     train_gru(data_name, data_loader, portfolios_data, label_name=f"GRU-feature{feature_size}", param=get_default_param())
    #     train_gru_ext(data_name, data_loader, portfolios_data, label_name=f"GRUExt-feature{feature_size}", param=get_default_param())
    # file_path = "Results/ModelCompare"
    # plot_portfolios(portfolios_data, data_loader.data_test_with_date, file_path, f"{data_name}-Compare-featureSize")
    # save_portfolios(portfolios_data, file_path, f"{data_name}-Compare-featureSize")
    #
    # # window_size
    # portfolios_data = {}
    # param = get_default_param()
    # for win_size in window_size_list:
    #     print(f"{'='*50} compare win_size {win_size}")
    #     param.WINDOW_SIZE = win_size
    #     train_mlp_windowed(data_name, data_loader, portfolios_data, label_name=f"MLP-window{win_size}", param=param)
    #     train_mlp_windowed_ext(data_name, data_loader, portfolios_data, label_name=f"MLPExt-window{win_size}", param=param)
    #     train_gru(data_name, data_loader, portfolios_data, label_name=f"GRU-window{win_size}", param=get_default_param())
    #     train_gru_ext(data_name, data_loader, portfolios_data, label_name=f"GRUExt-window{win_size}", param=get_default_param())
    # file_path = "Results/ModelCompare"
    # plot_portfolios(portfolios_data, data_loader.data_test_with_date, file_path, f"{data_name}-Compare-windowSize")
    # save_portfolios(portfolios_data, file_path, f"{data_name}-Compare-windowSize")
    #
    # # gamma
    # portfolios_data = {}
    # param = get_default_param()
    # for gamma in gamma_list:
    #     print(f"{'='*50} compare gamma {gamma}")
    #     param.GAMMA = gamma
    #     train_mlp_windowed(data_name, data_loader, portfolios_data, label_name=f"MLP-gamma{gamma}", param=param)
    #     train_mlp_windowed_ext(data_name, data_loader, portfolios_data, label_name=f"MLPExt-gamma{gamma}", param=param)
    #     train_gru(data_name, data_loader, portfolios_data, label_name=f"GRU-gamma{gamma}", param=get_default_param())
    #     train_gru_ext(data_name, data_loader, portfolios_data, label_name=f"GRUExt-gamma{gamma}", param=get_default_param())
    # file_path = "Results/ModelCompare"
    # plot_portfolios(portfolios_data, data_loader.data_test_with_date, file_path, f"{data_name}-Compare-Gamma")
    # save_portfolios(portfolios_data, file_path, f"{data_name}-Compare-Gamma")
    #
    # # n_step
    # portfolios_data = {}
    # param = get_default_param()
    # for n_step in n_step_list:
    #     print(f"{'='*50} compare n_step {n_step}")
    #     param.N_STEP = n_step
    #     train_mlp_windowed(data_name, data_loader, portfolios_data, label_name=f"MLP-nstep{n_step}", param=param)
    #     train_mlp_windowed_ext(data_name, data_loader, portfolios_data, label_name=f"MLPExt-nstep{n_step}", param=param)
    #     train_gru(data_name, data_loader, portfolios_data, label_name=f"GRU-nstep{n_step}", param=get_default_param())
    #     train_gru_ext(data_name, data_loader, portfolios_data, label_name=f"GRUExt-nstep{n_step}", param=get_default_param())
    # file_path = "Results/ModelCompare"
    # plot_portfolios(portfolios_data, data_loader.data_test_with_date, file_path, f"{data_name}-Compare-nStep")
    # save_portfolios(portfolios_data, file_path, f"{data_name}-Compare-nStep")
    #
    #
