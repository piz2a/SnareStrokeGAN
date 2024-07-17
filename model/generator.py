import torch
from torch import nn
import os


# Create embedding layer from trained data or initial data. If there is none, it creates one
def get_embedding_layer(directory):
    if os.path.exists(f'{directory}/trained_embedding.pt'):
        weight = torch.load(f'{directory}/trained_embedding.pt', weights_only=True)
    elif os.path.exists(f'{directory}/initial_embedding.pt'):
        weight = torch.load(f'{directory}/initial_embedding.pt', weights_only=True)
    else:
        return None
    return torch.nn.Embedding.from_pretrained(weight)


class GeneratorCell(nn.Module):
    def __init__(self, device, batch_size, n, embedding_layer):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.n = n
        self.conv1d_st1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2)
        self.relu_mt1 = nn.ReLU()
        self.linear_mt1 = nn.Linear(in_features=2*n, out_features=n)
        self.gru = nn.GRU(input_size=n, hidden_size=n, num_layers=1)
        self.linear_ht = nn.Linear(in_features=n, out_features=1)
        self.embedding_layer = embedding_layer
        self.relu_ct = nn.ReLU()
        self.linear_ct = nn.Linear(in_features=n+1, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, st1, ht1, zt, da_t):
        da_t = da_t.unsqueeze(1)
        Mt1 = torch.cat([self.conv1d_st1(st1).squeeze(), ht1], dim=1)  # b*2n
        Ht1 = self.linear_mt1(self.relu_mt1(Mt1)).unsqueeze(0)  # 1*b*n

        xt = torch.cat((zt, da_t), dim=1)  # b*n
        ht = self.gru(xt, Ht1)  # b*n

        # Sound sample
        it = self.linear(ht).squeeze()  # b*n -> b*1 -> b
        srt = self.embedding_layer(it)  # n

        # Conserve gate
        ct = self.sigmoid(self.linear_ct(self.relu_ct(torch.cat([ht, da_t], dim=1)))).squeeze()  # b

        # New sound (max 32767) tanh?
        st = ct * torch.cat([st1[:, da_t:], torch.zeros((self.batch_size, da_t)).to(self.device)], dim=1) + srt  # b*n

        return st, ht


class Generator(nn.Module):
    def __init__(self, device, batch_size, n, embedding_layer):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.n = n
        self.cell = GeneratorCell(self.device, batch_size, n, embedding_layer)

    def forward(self, z, alpha):  # z: b*k*(n-1), alpha: b*k
        ht = torch.zeros((self.batch_size, self.n)).to(self.device)
        st = torch.zeros((self.batch_size, self.n)).to(self.device)
        sum_s = torch.empty((self.batch_size, 0)).to(self.device)
        for i, zt in enumerate(z.permute(1, 0, 2)):  # k*b*(n-1). zt: b*(n-1)
            alpha_t1 = alpha[i-1] if i > 0 else 0
            da_t = alpha[i] - alpha_t1
            st, ht = self.cell(st, ht, zt, da_t)
            prev_sum_s = sum_s[:, :alpha[i]] if sum_s.size()[1] >= alpha[i] else torch.zeros((self.batch_size, alpha[i]))
            sum_s = torch.cat([prev_sum_s, st], dim=1)
        return sum_s  # b*l


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_layer = get_embedding_layer('../resources/embedding').to(device)
    print(embedding_layer)
    n = 24000
    batch_size = 2
    g = Generator(device, batch_size, n, embedding_layer).to(device)
    print(g)
    z = torch.randn((batch_size, 8, n-1))
    alpha = torch.tensor([[0, 3000, 6000, 9000, 12000, 15000, 18000, 21000], [0, 4500, 6000, 10500, 12000, 16500, 18000, 22500]])
    result = g(z, alpha)
    print(result)
