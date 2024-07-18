import torch
from torch import nn


class DiscriminatorCell(nn.Module):
    def __init__(self, frame_rate):
        super().__init__()
        self.frame_rate = frame_rate

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=frame_rate-2)  # (batch, 1, frame_rate) -> (batch, 1, 3)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # (batch, 1, 3) -> (batch, 32, 3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3)  # (batch, 32, 3) -> (batch, 128, 1)
        self.relu3 = nn.ReLU()
        self.linear = nn.Linear(in_features=129, out_features=256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1)

    def forward(self, interval, sound_fragment, ht1ct1):  # b, b*(frame_rate+1)
        x = self.relu2(self.conv2(self.conv1(sound_fragment.unsqueeze(1))))
        # print(x.size())
        x = self.conv3(x)
        # print(x.size(), interval.size())
        x = torch.cat([interval.unsqueeze(1), x.squeeze(2)], dim=1)  # Concatenate
        x = self.linear(x)
        output, (ht, ct) = self.lstm(x.unsqueeze(0), ht1ct1)
        return output, (ht, ct)


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
        batch_size = sound.shape[0]
        # print('batch:', batch_size)
        num_layers = 1
        ht = torch.zeros((num_layers, batch_size, 256)).to(self.device)
        ct = torch.zeros((num_layers, batch_size, 256)).to(self.device)
        for i, alpha_t in enumerate(alpha.permute(1, 0)):
            alpha_t1 = alpha[:, i-1] if i > 0 else 0
            sound_fragment = torch.empty((0, self.frame_rate)).to(self.device)
            # print(sound.shape)
            zero_padding = torch.zeros((batch_size, self.frame_rate // 2)).to(self.device)
            sound_with_padding = torch.cat([zero_padding, sound, zero_padding], dim=1)
            for b, alpha_tb in enumerate(alpha_t):
                # print(alpha_tb - self.frame_rate // 2, alpha_tb + self.frame_rate // 2)
                cropped_sound = sound_with_padding[b, alpha_tb : alpha_tb + self.frame_rate].unsqueeze(0)
                # print(cropped_sound.shape)
                sound_fragment = torch.cat([
                    sound_fragment,
                    cropped_sound
                ])
            # print('sound_fragment:', sound_fragment.shape)
            output, (ht, ct) = self.cell(alpha_t - alpha_t1, sound_fragment, (ht, ct))
        # print(ht.shape)
        return self.sigmoid(self.linear2(self.relu(self.linear1(ht))))
