import torch
from torch import nn


class DiscriminatorCell(nn.Module):
    def __init__(self, device, frame_rate):
        super().__init__()
        self.frame_rate = frame_rate

        self.gru = nn.GRU(input_size=frame_rate, hidden_size=3, device=device)  # (batch, frame_rate) -> (batch, 3)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1, device=device)  # (batch, 1, 3) -> (batch, 32, 3)
        self.relu2 = nn.ReLU(device)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, device=device)  # (batch, 32, 3) -> (batch, 16, 1)
        self.relu3 = nn.ReLU(device)
        self.linear = nn.Linear(in_features=17, out_features=10, device=device)
        self.lstm = nn.LSTMCell(input_size=10, hidden_size=10, device=device)

    def forward(self, interval, sound_fragment, ht1ct1):  # b, b*(frame_rate+1), (ht, ct)
        output, ht = self.gru(torch.abs(sound_fragment))
        # print(ht.size())
        x = self.relu2(self.conv2(ht.unsqueeze(1)))
        # print(x.size())
        x = self.conv3(x)
        # print(x.size(), interval.size())
        x = torch.cat([interval.unsqueeze(1), x.squeeze(2)], dim=1)  # Concatenate
        # print(x.size())
        x = self.linear(x)
        ht, ct = self.lstm(x, ht1ct1)
        return ht, ct


class Discriminator(nn.Module):
    def __init__(self, device, frame_rate):
        super().__init__()
        self.device = device
        self.frame_rate = frame_rate

        self.cell = DiscriminatorCell(device, frame_rate)
        self.linear1 = nn.Linear(in_features=10, out_features=32, device=device)
        self.relu = nn.ReLU(device)
        self.linear2 = nn.Linear(in_features=32, out_features=1, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, alpha, sound):
        batch_size = sound.shape[0]
        # print('batch:', batch_size)
        ht = torch.zeros((batch_size, 10)).to(self.device)
        ct = torch.zeros((batch_size, 10)).to(self.device)
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
            ht, ct = self.cell(alpha_t - alpha_t1, sound_fragment, (ht, ct))
        # print(ht.shape)
        return self.sigmoid(self.linear2(self.relu(self.linear1(ht))))
