# Importing DataLoaders for each model. These models include rule-based, vanilla DQN and encoder-decoder DQN.
import datetime
import math

import numpy as np

from DataLoader.DataLoader import YahooFinanceDataLoader
from DataLoader.DataForPatternBasedAgent import DataForPatternBasedAgent
from DataLoader.DataAutoPatternExtractionAgent import DataAutoPatternExtractionAgent
from DataLoader.DataSequential import DataSequential

from DeepRLAgent.MLPEncoder.Train import Train as SimpleMLP
from DeepRLAgent.SimpleCNNEncoder.Train import Train as SimpleCNN
from EncoderDecoderAgent.GRU.Train import Train as GRU
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

parser = argparse.ArgumentParser(description='DQN-Trader arguments')
parser.add_argument('--dataset-name', default="BTC-USD",
                    help='Name of the data inside the Data folder')
parser.add_argument('--nep', type=int, default=30,
                    help='Number of episodes')
parser.add_argument('--window_size', type=int, default=15,
                    help='Window size for sequential models')
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

    'GOOGL': YahooFinanceDataLoader('GOOGL',
                                    split_point='2018-01-01',
                                    load_from_file=True),

    'AAPL': YahooFinanceDataLoader('AAPL',
                                   split_point='2018-01-01',
                                   begin_date='2010-01-01',
                                   end_date='2020-08-24',
                                   load_from_file=True),

    'DJI': YahooFinanceDataLoader('DJI',
                                  split_point='2016-01-01',
                                  begin_date='2009-01-01',
                                  end_date='2018-09-30',
                                  load_from_file=True),

    'S&P': YahooFinanceDataLoader('S&P',
                                  split_point=2000,
                                  end_date='2018-09-25',
                                  load_from_file=True),

    'AMD': YahooFinanceDataLoader('AMD',
                                  split_point=2000,
                                  end_date='2018-09-25',
                                  load_from_file=True),

    'GE': YahooFinanceDataLoader('GE',
                                 split_point='2015-01-01',
                                 load_from_file=True),

    'KSS': YahooFinanceDataLoader('KSS',
                                  split_point='2018-01-01',
                                  load_from_file=True),

    'HSI': YahooFinanceDataLoader('HSI',
                                  split_point='2015-01-01',
                                  load_from_file=True),

    'AAL': YahooFinanceDataLoader('AAL',
                                  split_point='2018-01-01',
                                  load_from_file=True),

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
            # # 'DQN-pattern': {},
            # # 'DQN-vanilla': {},
            # # 'DQN-candlerep': {},
            # 'DQN-windowed': {},
            # 'DQN-windowed-ext': {},
            # # 'MLP-pattern': {},
            # # 'MLP-vanilla': {},
            # # 'MLP-candlerep': {},
            'MLP-windowed': {},
            # 'MLP-windowed-ext': {},
            # # 'CNN1d': {},
            # 'CNN2d': {},
            # 'CNN2d-ext': {},
            'GRU': {},
            # 'GRU-ext': {},
            # # 'Deep-CNN': {},
            # 'CNN-GRU': {},
            # 'CNN-GRU-ext': {},
            # # 'CNN-ATTN': {},
            'BAH': {},
            'MM-DQN': {},
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

    def load_stored_data(self, to_use):
        self.test_portfolios['MLP-windowed'] = load_pkl(f"Results/portfolios-use/{to_use[self.dataset_name]['l']}")
        self.test_portfolios['GRU'] = load_pkl(f"Results/portfolios-use/{to_use[self.dataset_name]['r']}")
        self.test_portfolios['MM-DQN'] = load_pkl(f"Results/portfolios-use/{to_use[self.dataset_name]['my']}")
        # print(f"{dataset_name}: total return, volatility, sharpe ratio")
        print("")
        for label in ['MLP-windowed', 'GRU', 'MM-DQN']:
            portfolios = self.test_portfolios[label]
            rate_of_return = self.rate_of_return(portfolios)
            print(f"{round(self.total_return(portfolios), 2)}, {round(self.calculate_daily_return_volatility(rate_of_return), 5)}, {round(self.sharp_ratio(rate_of_return), 5)}")

    def total_return(self, portfolio_value):
        return (portfolio_value[-1] - 1000) / 1000 * 100

    def rate_of_return(self, portfolio_value):
        return [(portfolio_value[p + 1] - portfolio_value[p]) / portfolio_value[p] for p in range(len(portfolio_value) - 1)]

    def calculate_daily_return_volatility(self, rate_of_return):
        # 计算平均收益率
        mean_return = np.mean(rate_of_return)
        T = len(rate_of_return)
        volatility = np.sqrt(sum([(r - mean_return) ** 2 for r in rate_of_return]) / (T - 1))

        return volatility

    def sharp_ratio(self, rate_of_return):
        return np.mean(rate_of_return) / np.std(rate_of_return)

    def evaluate_models(self):
        # self.test_portfolios['DQN-windowed'] = self.dqn_windowed.test().get_daily_portfolio_value()
        # self.test_portfolios['DQN-windowed-ext'] = self.dqn_windowed_ext.test().get_daily_portfolio_value()
        # self.test_portfolios['MLP-windowed'] = self.mlp_windowed.test().get_daily_portfolio_value()
        # self.test_portfolios['MLP-windowed-ext'] = self.mlp_windowed_ext.test().get_daily_portfolio_value()
        # self.test_portfolios['CNN2d'] = self.cnn2d.test().get_daily_portfolio_value()
        # self.test_portfolios['CNN2d-ext'] = self.cnn2d_ext.test().get_daily_portfolio_value()
        # self.test_portfolios['GRU'] = self.gru.test().get_daily_portfolio_value()
        # self.test_portfolios['GRU-ext'] = self.gru_ext.test().get_daily_portfolio_value()
        # self.test_portfolios['CNN-GRU'] = self.cnn_gru.test().get_daily_portfolio_value()
        # self.test_portfolios['CNN-GRU-ext'] = self.cnn_gru_ext.test().get_daily_portfolio_value()
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
            print(f"=========== {(name + ' '*20)[:20]} Rate: {round(left_val/right_val, 2)}      {'**' if right_val <= 0 else ''}")

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
        # self.save_portfolios()
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


if __name__ == '__main__':
    start_time = dt.datetime.now()
    print(f"--------- total start time: {start_time}")

    gamma_list = [0.9, 0.8, 0.7]
    batch_size_list = [16, 64, 256]
    replay_memory_size_list = [16, 64, 256]

    n_step = 8  # 多步时间差分
    window_size = args.window_size  # 默认值设置为15，根据论文里来的
    dataset_name = args.dataset_name
    n_episodes = args.nep  # 默认30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch version: {torch.__version__}")
    print(f"cal device: {device}")

    feature_size = 64  # 编码器的输出大小
    target_update = 5  # 5 次后更新目标网络

    # gamma_default = 0.9
    # batch_size_default = 16
    # replay_memory_size_default = 32

    # 论文中的默认值
    gamma_default = 0.9  # 奖励衰减
    batch_size_default = 10  # mini-batch 方法的 batch 大小
    replay_memory_size_default = 20  # 经验回放组大小

    data_list = [
        '000651',
        '000725',
        '000858',
        '600030',
        '600036',
        '600276',
        '600519',
        '600887',
        '600900',
        '601398',
    ]

    to_use = load_pkl("./to_use.pkl")

    for dataset_name in data_list:
        # print(str(datetime.datetime.now()) + ' : ' + dataset_name)
        # print(dataset_name)
        run = SensitivityRun(
            dataset_name,
            gamma_default,
            batch_size_default,
            replay_memory_size_default,
            feature_size,
            target_update,
            n_episodes,
            n_step,
            window_size,
            device,
            evaluation_parameter='ModelCompare',
            transaction_cost=0,
            alpha_extension=[
                'alpha_019',
                'alpha_026',
                'alpha_024',
                'alpha_074',
                'alpha_081',
                'alpha_032',
                'alpha_050',
                'alpha_099',
                'alpha_088',
                'alpha_061',
            ]
        )
        run.reset()
        run.load_stored_data(to_use)
        # run.train()
        run.evaluate_models()
        # run.print_evaluate_models()
        run.save_models_compare()
