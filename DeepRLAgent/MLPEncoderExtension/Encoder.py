import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Encoder(nn.Module):

    def __init__(self, num_classes, state_size, ext_size):
        """

        :param state_size: we give OHLC as input to the network
        :param action_length: Buy, Sell, Idle
        """
        super(Encoder, self).__init__()

        print(f"init Encoder: num_classes: {num_classes}, state_size: {state_size}")
        self.encoder = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.BatchNorm1d(128),
            # nn.Linear(128, 256),
            # nn.BatchNorm1d(256),
            nn.Linear(128, num_classes)
        )
        self.ext_encoder = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.BatchNorm1d(128),
            # nn.Linear(128, 256),
            # nn.BatchNorm1d(256),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
