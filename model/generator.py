import torch
from torch import nn

# m = 49, n = 900, k = 44100
# m = 49, n = 450, k = 22050
# k(sound sample length). ceil(k/m) = n, n-1 < k/m <= n, m(n-1) < k <= mn, 44051 < k <= 44100


class GeneratorCell(nn.Module):
    def __init__(self, batch_size, m, n, k, embedding_layer):
        super().__init__()
        self.batch_size = batch_size
        assert m * (n-1) < k <= m * n
        self.m, self.n, self.k = m, n, k

        self.conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(m+1, 1), stride=1, padding=0)

        # Weights and biases used for GRU
        self.W_xr = nn.Parameter(torch.randn(2, 2))
        self.W_hr = nn.Parameter(torch.randn(2, 2))
        self.b_r = nn.Parameter(torch.randn(2))

        self.W_xu = nn.Parameter(torch.randn(2, 2))
        self.W_hu = nn.Parameter(torch.randn(2, 2))
        self.b_u = nn.Parameter(torch.randn(2))

        self.W_xn = nn.Parameter(torch.randn(2, 2))
        self.W_hn = nn.Parameter(torch.randn(2, 2))
        self.b_n = nn.Parameter(torch.randn(2))

        # Layers for creating waves
        self.linear = nn.Linear(in_features=n*2, out_features=n)
        self.embedding_layer = embedding_layer

        self.qt_padding = n * m - k

    def forward(self, qt1, ht1, zt, alpha_t):
        Mt1 = torch.cat([qt1, ht1.unsqueeze(1)], dim=1)  # b*(m+1)*n*2
        Ht1 = self.conv(Mt1.permute(0, 3, 1, 2)).squeeze()  # Mt1 -(permute)-> b*2*(m+1)*n -(CNN)(squeeze)-> b*n*2
        xt = torch.stack((zt, alpha_t), dim=1).permute(0, 2, 1)  # b*n*2

        # GRU
        rt = torch.sigmoid(torch.matmul(xt, self.W_xr) + torch.matmul(Ht1, self.W_hr) + self.b_r)
        ut = torch.sigmoid(torch.matmul(xt, self.W_xu) + torch.matmul(Ht1, self.W_hu) + self.b_u)
        nt = torch.tanh(torch.matmul(xt, self.W_xn) + torch.matmul(rt * Ht1, self.W_hn) + self.b_n)
        ht = ut * ht1 + (1 - ut) * nt  # b*n*2

        # Sound sample indexes
        it = self.linear(ht.flatten(start_dim=1))  # b*2n -> b*n

        # Sound sample data
        srt = self.embedding_layer(it) * alpha_t.unsqueeze(-1).unsqueeze(-1)  # embedding result * alpha: b*n*k*2

        # Overlapping sounds
        f_srt = torch.zeros((self.batch_size, self.n+self.k-1, 2))
        for i, srti in enumerate(srt.permute(1, 0, 2, 3)):  # b*n*k*2 => n*b*k*2
            f_srt += torch.cat([torch.zeros((self.batch_size, i, 2)), srti], dim=1)
        # f_srt: b*(n+k-1)*2

        st = qt1[:, 0] + f_srt[:, :self.n]  # b*n*2
        qt = torch.unflatten(
            (
                torch.cat([qt1[:, 1:], torch.zeros(st.shape).unsqueeze(1)], dim=1) +
                torch.cat([f_srt[:, self.n:self.n+self.k-1], torch.zeros((self.batch_size, self.qt_padding, 2))])  # b*(k+padding)*2
            ), 1, (self.m, self.n)
        )  # b*m*n*2

        return st, qt, ht


class Generator(nn.Module):
    def __init__(self, device, batch_size, m, n, k, embedding_layer):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.m, self.n, self.k = m, n, k
        self.cell = GeneratorCell(batch_size, m, n, k, embedding_layer)

    def forward(self, z, alpha):  # z, alpha: b*l*n
        qt = torch.zeros((self.batch_size, self.m, self.n, 2)).to(self.device)
        ht = torch.zeros((self.batch_size, self.n, 2)).to(self.device)
        sum_h = torch.empty((self.batch_size, 0, 2)).to(self.device)
        for i, zt in enumerate(z.permute(1, 0, 2)):  # l*b*n
            alpha_t = alpha[i]
            qt, ht = self.cell(qt, ht, zt, alpha_t)
            sum_h = torch.cat((sum_h, ht), dim=1)
        # z, alpha 끝나도 남은 소리 generating 하기(input에 0 패딩)
        return sum_h  # b*l*2
