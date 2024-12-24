import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FCmodel(nn.Module):
    def __init__(self, input_size, output_size, hdim=100):
        super().__init__()

        self.fc = torch.nn.Sequential(nn.Linear(input_size, hdim), nn.ReLU(), nn.Linear(hdim, hdim), nn.ReLU(), nn.Linear(hdim, output_size))

    def forward(self, img):
        x = img.view(img.size(0), -1)
        y = self.fc(x)

        return y


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        # nn.BatchNorm2d(out_channels, track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class CNNCifar(nn.Module):
    def __init__(self, output_size, hidden_dim=64):
        super(CNNCifar, self).__init__()
        in_channels = 3
        num_classes = output_size

        hidden_size = hidden_dim  # 64 for default

        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            # conv3x3(hidden_size, hidden_size)
        )

        self.linear0 = nn.Linear(hidden_size * 2 * 2 * 4, hidden_size * 2 * 2)  # added
        self.linear1 = nn.Linear(hidden_size * 2 * 2, hidden_size * 2)  # added
        self.linear2 = nn.Linear(hidden_size * 2, num_classes)

        # self.linear = nn.Linear(hidden_size*2*2, num_classes)

        self.apply(init_weights)

    def forward(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))

        features = torch.nn.functional.leaky_relu(self.linear0(features), negative_slope=0.1)  # added
        features = torch.nn.functional.leaky_relu(self.linear1(features), negative_slope=0.1)  # added
        logits = self.linear2(features)

        return logits

    def extract_features(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))

        return features





@torch.no_grad()
def init_weights(m):
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
            nn.init.xavier_normal_(m.weight)
            # m.weight.fill_(0.01)
        if hasattr(m, "bias") and m.bias is not None:
            # m.bias.fill_(0.001)
            nn.init.normal_(m.bias, mean=0, std=1e-3)
            # m.bias.fill_(0.01)

    if type(m) == nn.Conv2d:
        if hasattr(m, "weight"):
            # gain = torch.nn.init.calculate_gain('relu')
            # nn.init.xavier_normal_(m.weight, gain=gain)
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.xavier_normal_(m.weight)
            # m.weight.fill_(0.01)
        if hasattr(m, "bias") and m.bias is not None:
            #    m.bias.fill_(0.01)
            nn.init.normal_(m.bias, mean=0, std=1e-3)

