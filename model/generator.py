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
        self.sigmoid_it = nn.Sigmoid()
        self.embedding_layer = embedding_layer
        self.relu_ct = nn.ReLU()
        self.linear_ct = nn.Linear(in_features=n+1, out_features=1)
        self.sigmoid_ct = nn.Sigmoid()

    def forward(self, st1, ht1, zt, da_t):
        da_t = da_t.unsqueeze(1)
        Mt1 = torch.cat([self.conv1d_st1(st1.unsqueeze(1)).squeeze(1), ht1], dim=1)  # b*2n
        Ht1 = self.linear_mt1(self.relu_mt1(Mt1)).unsqueeze(0)  # 1*b*n

        # print(f'zt: {zt.size()}, da_t: {da_t.size()}')
        xt = torch.cat((zt, da_t), dim=1).unsqueeze(0)  # 1*b*n
        # print(f'xt: {xt.size()}, Ht1: {Ht1.size()}')
        output, ht = self.gru(xt, Ht1)  # 1*b*n
        ht = ht.squeeze(0)
        # print(output == ht)

        # Sound sample
        it = ((self.embedding_layer.weight.size()[0] - 1) *  # 284 - 1
              self.sigmoid_it(self.linear_ht(ht.squeeze(0)).squeeze(1))).long()  # 1*b*n -> b*n -> b*1 -> b
        # print(f'ht: {ht.size()}, it: {it.size()}')
        print('it value:', it)
        srt = self.embedding_layer(it)  # b*n

        # Conserve gate
        ct = self.sigmoid_ct(self.linear_ct(self.relu_ct(torch.cat([ht, da_t], dim=1)))).squeeze()  # b

        # New sound (max 32767   이거 해야 함) tanh?
        st1_fragment = torch.empty((0, self.n)).to(self.device)  # batch의 각 data마다 길이가 다름
        for b, st1_b in enumerate(st1):
            print("frag", da_t.size(), st1.size()[1] - da_t.size()[1] + 1, st1_b[da_t[b]:].size())
            zeros = torch.zeros(da_t[b]).to(self.device)
            st1_fragment = torch.cat((
                st1_fragment,
                torch.cat([st1_b[da_t[b]:], zeros]).unsqueeze(0) * ct[b]
            ))
        print(f'ct: {ct}, st1_fragment: {st1_fragment.size()}, zeros: {zeros.size()}')
        st = torch.cat([st1_fragment, zeros], dim=1) + srt  # b*n

        return st, ht


class Generator(nn.Module):
    def __init__(self, device, batch_size, n, frame_count, embedding_layer):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.n = n
        self.frame_count = frame_count
        self.cell = GeneratorCell(self.device, batch_size, n, embedding_layer)

    def forward(self, z, alpha):  # z: b*k*(n-1), alpha: b*k
        ht = torch.zeros((self.batch_size, self.n)).to(self.device)
        st = torch.zeros((self.batch_size, self.n)).to(self.device)
        sum_s = torch.zeros((self.batch_size, self.frame_count)).to(self.device)
        for i, zt in enumerate(z.permute(1, 0, 2)):  # k*b*(n-1). zt: b*(n-1)
            alpha_t1 = alpha[:, i-1] if i > 0 else 0
            da_t = alpha[:, i] - alpha_t1
            st, ht = self.cell(st, ht, zt, da_t)
            print(f"max(st): {torch.max(st)}, ht: {ht.size()}")
            for b, alpha_ib in enumerate(alpha[:, i]):
                print(f"alpha_ib: {sum_s[b, :alpha_ib].size()}, st: {st[b].size()}, zeros: {torch.zeros(sum_s.size()[1] - alpha_ib - self.n).size()}")
                sum_s[b] = torch.cat([sum_s[b, :alpha_ib], st[b], torch.zeros((sum_s.size()[1] - alpha_ib - self.n,)).to(self.device)])
        return sum_s  # b*l


if __name__ == '__main__':
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    embedding_layer = get_embedding_layer('../resources/embedding').to(device)
    print(embedding_layer)
    n = 14676
    batch_size = 2
    frame_count = 24000
    g = Generator(device, batch_size, n, frame_count, embedding_layer).to(device)
    print(g)
    z = torch.randn((batch_size, 8, n-1))
    alpha = torch.LongTensor([[0, 3000, 6000, 9000, 12000, 15000, 18000, 21000], [0, 4500, 6000, 10500, 12000, 16500, 18000, 22500]])
    result = g(z, alpha)
    print(result)
