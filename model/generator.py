import torch
from torch import nn
import os


DEBUG = False


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

    def __init__(self, device, batch_size, n, m, embedding_layer):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.n = n
        self.m = m
        self.conv1d_st1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2, device=device)
        self.conv1d_st1_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=round((n-2)/(self.m-1)), padding=0, device=device)
        self.relu_mt1 = nn.ReLU(device)
        self.conv1d_mt1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, device=device)
        self.conv1d_xt = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=round((n-2)/(self.m-1)), padding=0, device=device)
        self.gru = nn.GRU(input_size=self.m, hidden_size=self.m, num_layers=1, device=device)
        self.linear_ht = nn.Linear(in_features=self.m, out_features=1, device=device)
        self.sigmoid_it = nn.Sigmoid().to(device)
        self.embedding_layer = embedding_layer
        embedding_layer.requires_grad = False
        self.relu_ct = nn.ReLU(device)
        self.linear_ct = nn.Linear(in_features=self.m+1, out_features=1, device=device)
        self.sigmoid_ct = nn.Sigmoid().to(device)

    def forward(self, st1, ht1, zt, da_t):
        da_t = da_t.unsqueeze(1)
        Mt1_0 = self.conv1d_st1_2(self.conv1d_st1(st1.unsqueeze(1))).squeeze(1)
        Mt1 = torch.cat([Mt1_0, ht1], dim=1)  # b*2m
        Mt1_2 = self.conv1d_mt1(self.relu_mt1(Mt1).unsqueeze(1))  # b*m -> b*1*m
        if DEBUG: print(f'Mt1_0: {Mt1_0.shape} Mt1: {Mt1.shape} Mt1_2: {Mt1_2.shape}')
        Ht1 = Mt1_2.permute(1, 0, 2)  # b*1*m -> 1*b*m

        if DEBUG: print(f'zt: {zt.size()}, da_t: {da_t.size()}')
        xt = self.conv1d_xt(torch.cat((zt, da_t), dim=1).unsqueeze(1)).permute(1, 0, 2)  # b*1*n -> b*1*m -> 1*b*m
        if DEBUG: print(f'xt: {xt.size()}, Ht1: {Ht1.size()}')
        output, ht = self.gru(xt, Ht1)  # 1*b*n
        ht = ht.squeeze(0)
        if DEBUG: print(output == ht)

        # Sound sample
        it = ((self.embedding_layer.weight.size()[0] - 1) *  # 284 - 1
              self.sigmoid_it(self.linear_ht(ht.squeeze(0)).squeeze(1))).long()  # 1*b*n -> b*n -> b*1 -> b
        if DEBUG: print(f'ht: {ht.size()}, it: {it.size()}')
        if DEBUG: print('it value:', it)
        with torch.no_grad():
            srt = self.embedding_layer(it)  # b*n

        # Conserve gate
        ct = self.sigmoid_ct(self.linear_ct(self.relu_ct(torch.cat([ht, da_t], dim=1)))).squeeze()  # b

        # New sound (max 32767   이거 해야 함) tanh?
        st1_and_zeros = torch.empty((0, self.n)).to(self.device)  # b*n. the sound length of each data of a batch varies
        for b, st1_b in enumerate(st1):
            if DEBUG: print("frag", da_t.size(), st1.size()[1] - da_t.size()[1] + 1, st1_b[da_t[b]:].size())
            zeros = torch.zeros(min(da_t[b], self.n)).to(self.device)   # da_t[b] > n이면 이전과 현재 스트로크의 간격이 넓어 남은 소리가 없는 경우.
            if DEBUG: print(st1_b.size(), st1_b[da_t[b]:].size(), zeros.size())
            st1_and_zeros = torch.cat((
                st1_and_zeros,
                torch.cat([st1_b[da_t[b]:], zeros]).unsqueeze(0) * ct[b]
            ))
        if DEBUG: print(f'ct: {ct}, st1_fragment: {st1_and_zeros.size()}, zeros: {zeros.size()}')
        st = st1_and_zeros + srt  # b*n

        return st, ht


class Generator(nn.Module):

    m = 10

    def __init__(self, device, batch_size, n, frame_count, embedding_layer):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.n = n
        self.frame_count = frame_count
        self.cell = GeneratorCell(self.device, batch_size, n, self.m, embedding_layer)

    def forward(self, z, alpha):  # z: b*k*(n-1), alpha: b*k
        ht = torch.zeros((self.batch_size, self.m)).to(self.device)
        st = torch.zeros((self.batch_size, self.n)).to(self.device)
        sum_s = torch.zeros((self.batch_size, self.frame_count)).to(self.device)
        for i, zt in enumerate(z.permute(1, 0, 2)):  # k*b*(n-1). zt: b*(n-1)
            alpha_t1 = alpha[:, i-1] if i > 0 else 0
            da_t = alpha[:, i] - alpha_t1
            if DEBUG: print('da_t:', da_t)
            st, ht = self.cell(st, ht, zt, da_t)
            if DEBUG: print(f"max(st): {torch.max(st)}, ht: {ht.size()}")
            for b, alpha_ib in enumerate(alpha[:, i]):
                padding_length = self.frame_count - alpha_ib - self.n
                if DEBUG: print(f"alpha_ib: {sum_s[b, :alpha_ib].size()}, st: {st[b].size()}, padding_length: {padding_length}")
                if padding_length >= 0:
                    sum_s[b] = torch.cat([sum_s[b, :alpha_ib], st[b], torch.zeros(padding_length).to(self.device)])
                elif alpha_ib < self.frame_count:
                    sum_s[b] = torch.cat([sum_s[b, :alpha_ib], st[b, :self.frame_count - alpha_ib]])
        return sum_s  # b*l


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    DEBUG = True

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    embedding_layer = get_embedding_layer('../resources/embedding').to(device)
    print(embedding_layer)
    n = 14676
    batch_size = 2
    frame_count = 48000
    g = Generator(device, batch_size, n, frame_count, embedding_layer).to(device)
    print(g)
    z = torch.randn((batch_size, 8, n-1))
    alpha = torch.LongTensor([[0, 3000, 6000, 9000, 12000, 15000, 18000, 21000], [0, 4500, 6000, 10500, 12000, 16500, 18000, 22500]])
    with torch.no_grad():
        result = g(z, alpha)
        print(result)
        plt.plot(list(range(frame_count)), result[1])
        plt.show()
