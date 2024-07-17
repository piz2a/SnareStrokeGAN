import torch
from torch import nn


class DiscriminatorCell(nn.Module):
    def __init__(self, frame_rate):
        super().__init__()
        self.frame_rate = frame_rate

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=frame_rate-1)  # (batch, 1, frame_rate) -> (batch, 1, 3)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # (batch, 1, 3) -> (batch, 32, 3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3)  # (batch, 32, 3) -> (batch, 128, 1)
        self.relu3 = nn.ReLU()
        self.linear = nn.Linear(in_features=129, out_features=256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1)

    def forward(self, interval, sound_fragment):  # b, b*(frame_rate+1)
        x = self.conv3(self.relu2(self.conv2(self.conv1(sound_fragment))))
        x = torch.cat([interval.unsqueeze(1), x], dim=1)  # Concatenate
        x = self.linear(x)
        output, ht = self.lstm(x)
        return output, ht


class Discriminator(nn.Module):
    def __init__(self, device, frame_rate):
        super().__init__()
        self.device = device
        self.frame_rate = frame_rate

        self.cell = DiscriminatorCell(frame_rate)
        self.linear1 = nn.Linear(in_features=256, out_features=32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=32, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, alpha, sound):
        ht = torch.zeros(0).to(self.device)
        for i, alpha_t in enumerate(alpha.permute(1, 0)):
            alpha_t1 = alpha[:, i-1] if i > 0 else 0
            output, ht = self.cell(alpha_t - alpha_t1, sound[:, alpha_t - self.frame_rate // 2 : alpha_t + self.frame_rate // 2])
        return self.sigmoid(self.linear2(self.relu(self.linear1(ht))))
