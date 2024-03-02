import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, state_size, ext_size, hidden_size, device):
        """
        :param input_size: 5 which is OHLC + trend
        """
        print(f"init EncoderRNN: state_size: {state_size}, ext_size: {ext_size}, hidden_size: {hidden_size}")
        # EncoderRNN: input_size: 14, hidden_size: 128

        super(EncoderRNN, self).__init__()
        self.device = device
        self.state_cnt = state_size
        self.ext_cnt = ext_size
        self.hidden_size = hidden_size
        self.state_gru = nn.GRU(self.state_cnt, hidden_size)
        self.ext_gru = nn.GRU(self.ext_cnt, hidden_size)
        # self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        """
        :param x: if the input x is a batch, its size is of the form [window_size, batch_size, input_size]
        thus, the output of GRU would be of shape [window_size, batch_size, hidden_size].
        e.g. output[:, 0, :] is the output sequence of the first element in the batch.
        The hidden is of the shape [1, batch_size, hidden_size]
        """

        # x.shape: 15, 10, 4 (window_size, batch_size, 特征数量)
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        state_hidden = self.initHidden(x.shape[1])
        ext_hidden = self.initHidden(x.shape[1])

        state_output, state_hidden = self.state_gru(x[:, :, :self.state_cnt], state_hidden)
        ext_output, ext_hidden = self.ext_gru(x[:, :, self.state_cnt:], ext_hidden)

        return torch.cat((state_output, ext_output), dim=2), torch.cat((state_hidden, ext_hidden), dim=2)

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)
