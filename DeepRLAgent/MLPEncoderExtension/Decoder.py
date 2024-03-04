import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, num_classes, action_length=3):
        """

        :param state_length: we give OHLC as input to the network
        :param action_length: Buy, Sell, Idle
        """
        super(Decoder, self).__init__()

        # print(f"init Decoder: num_classes: {num_classes}, action_length: {action_length}")

        self.state_policy_network = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.BatchNorm1d(128),
            # nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            # nn.LeakyReLU(),
            nn.Linear(256, action_length),
            nn.Softmax(),
        )

        self.ext_policy_network = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.BatchNorm1d(128),
            # nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            # nn.LeakyReLU(),
            nn.Linear(256, action_length),
            nn.Softmax(),
        )

    def forward(self, x):
        # print(f"decoder x len: {len(x)}, {x[0].shape}, {x[1].shape}")
        state_out = self.state_policy_network(x[0])
        ext_out = self.ext_policy_network(x[1])

        # print(f"state_out {state_out}")
        # print(f"ext_out {ext_out}")

        return state_out + ext_out * 0.5
