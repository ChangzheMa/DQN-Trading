import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # c is the context
        state_out, ext_out, gru_state_out, gru_state_hidden, gru_ext_out, gru_ext_hidden = self.encoder(x)
        output = self.decoder([state_out, ext_out, gru_state_out, gru_state_hidden, gru_ext_out, gru_ext_hidden])
        return output
