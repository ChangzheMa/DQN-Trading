import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Encoder(nn.Module):

    def __init__(self, num_classes, state_size, ext_size, window_size):
        """

        :param state_size: we give OHLCT as input to the network, 5 * window_size
        :param ext_size: 补充参数的数量, 指标个数 * window_size
        :param window_size: 窗口长度
        :param action_length: Buy, Sell, Idle
        """
        super(Encoder, self).__init__()

        print(f"init Encoder: num_classes: {num_classes}, state_size: {state_size}")

        self.state_cnt = round(state_size/window_size)
        self.ext_cnt = round(ext_size/window_size)
        self.window_size = window_size

        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes),
        )
        self.ext_encoder = nn.Sequential(
            nn.Linear(ext_size, max(ext_size * 4, 128)),
            nn.BatchNorm1d(max(ext_size * 4, 128)),
            nn.LeakyReLU(),
            nn.Linear(max(ext_size * 4, 128), 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes),
        )
        self.combine_encoder = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes * 4),
            nn.LeakyReLU(),
            nn.Linear(num_classes * 4, num_classes * 4),
            nn.LeakyReLU(),
            nn.Linear(num_classes * 4, num_classes),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.window_size, self.state_cnt + self.ext_cnt)
        state_x = x_reshaped[:, :, :self.state_cnt].contiguous().view(batch_size, -1)
        ext_x = x_reshaped[:, :, self.state_cnt:].contiguous().view(batch_size, -1)

        state_out = self.state_encoder(state_x)
        ext_out = self.ext_encoder(ext_x)

        return self.combine_encoder(torch.cat((state_out, ext_out), dim=-1))
