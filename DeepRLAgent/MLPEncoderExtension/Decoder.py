import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, num_classes, hidden_size, action_length=3, weight=(1, 1, 1, 1)):
        """

        :param state_length: we give OHLC as input to the network
        :param action_length: Buy, Sell, Idle
        """
        super(Decoder, self).__init__()

        self.weight = weight

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

        # self.state_policy_weight = nn.Sequential(
        #     nn.Linear(num_classes, 128),
        #     nn.BatchNorm1d(128),
        #     nn.Linear(128, 1),
        #     nn.Sigmoid(),
        # )

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

        # self.ext_policy_weight = nn.Sequential(
        #     nn.Linear(num_classes, 128),
        #     nn.BatchNorm1d(128),
        #     nn.Linear(128, 1),
        #     nn.Sigmoid(),
        # )

        self.gru_state_policy_network = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            # nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            # nn.LeakyReLU(),
            nn.Linear(256, action_length),
            nn.Softmax(),
        )

        # self.gru_state_policy_weight = nn.Sequential(
        #     nn.Linear(num_classes, 128),
        #     nn.BatchNorm1d(128),
        #     nn.Linear(128, 1),
        #     nn.Sigmoid(),
        # )

        self.gru_ext_policy_network = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            # nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            # nn.LeakyReLU(),
            nn.Linear(256, action_length),
            nn.Softmax(),
        )

        # self.gru_ext_policy_weight = nn.Sequential(
        #     nn.Linear(num_classes, 128),
        #     nn.BatchNorm1d(128),
        #     nn.Linear(128, 1),
        #     nn.Sigmoid(),
        # )

    def forward(self, x):
        # for i in range(len(x)):
        #     print(f"x[{i}].shape: {x[i].shape}")
        # print(f"decoder x len: {len(x)}, {x[0].shape}, {x[1].shape}")
        state_out = self.state_policy_network(x[0])
        # state_out_weight = self.state_policy_weight(x[0])

        ext_out = self.ext_policy_network(x[1])
        # ext_out_weight = self.ext_policy_weight(x[1])

        state_hidden = x[3].squeeze().unsqueeze(0) if len(x[3].squeeze().shape) < 2 else x[3].squeeze()
        gru_state_out = self.gru_state_policy_network(state_hidden).squeeze()
        # gru_state_out_weight = self.gru_state_policy_weight(state_hidden)

        ext_hidden = x[5].squeeze().unsqueeze(0) if len(x[5].squeeze().shape) < 2 else x[5].squeeze()
        gru_ext_out = self.gru_ext_policy_network(ext_hidden)
        # gru_ext_out_weight = self.gru_ext_policy_weight(ext_hidden)



        # print(f"state_out: {state_out.shape}, ext_out: {ext_out.shape}, "
        #       f"gru_state_out: {gru_state_out.shape}, gru_ext_out: {gru_ext_out.shape}, ")

        # print(f"state_out {state_out}")
        # print(f"ext_out {ext_out}")
        # print(f"\nstate_out    : {self.findMax(state_out)}"
        #       f"\next_out      : {self.findMax(ext_out)}"
        #       f"\ngru_state_out: {self.findMax(gru_state_out)}"
        #       f"\ngru_ext_out  : {self.findMax(gru_ext_out)}")

        return (state_out * self.weight[0] + ext_out * self.weight[1] +
                gru_state_out * self.weight[2] + gru_ext_out * self.weight[3])

    def findMax(self, x):
        if len(x.shape) < 2:
            x = x.unsqueeze(dim=0)

        max_indices = torch.argmax(x, dim=1)
        return F.one_hot(max_indices, num_classes=x.shape[1]).to(dtype=x.dtype)
