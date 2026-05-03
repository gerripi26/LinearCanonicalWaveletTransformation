from einops import rearrange
import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
import torch.fft as fft
from torch import pi as pi
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LCWT(nn.Module):
    def __init__(self):
        super().__init__()

        # parameters
        self.n_classes = 5
        self.n_heads = 4
        self.N = 186
        self.register_buffer('n', torch.arange(-self.N // 2, self.N // 2))

        n_scales = 64
        a_min = 0.01
        a_max = 1
        init_scales = torch.exp(torch.linspace(torch.log(torch.tensor(a_min)), torch.log(torch.tensor(a_max)), n_scales))
        init_scales = torch.log(torch.exp(init_scales) - 1)  # softplus inverse
        self.a_param = nn.Parameter(init_scales)

        # lct matrix
        self.alpha = nn.Parameter(torch.tensor([-1.0, 0.0, 1.0, 2.0])[:, None, None])
        self.beta_param = nn.Parameter(torch.tensor([0.0, 1.0, 2.0, 3.0])[:, None, None])
        self.gamma = nn.Parameter(torch.tensor([1.0, 0.0, 1.0, -1.0])[:, None, None])

        # mother
        f0_init = 0.1
        sigma_init = 1
        self.f0 = nn.Parameter(torch.ones(self.n_heads,)[:, None, None] * f0_init)
        self.sigma = nn.Parameter(torch.ones(self.n_heads,)[:, None, None] * sigma_init)

        # resnet
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=self.n_heads,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.n_classes)

    # LCT
    def lct(self, x, m):
        x_shift = fft.ifftshift(x)
        X_m_shift = torch.exp(1j*pi * self.alpha * m**2) * fft.fft(x_shift * torch.exp(1j * self.gamma * self.n**2))
        X_m = fft.fftshift(X_m_shift)
        return X_m

    def forward(self, x, plot=False):
        # Dims: (Batch Head a n)
        x = x[:, None, None, :]
        beta = F.softplus(self.beta_param, beta=1)
        dm = 1 / (self.N * beta)                     # (H 1 1)
        m = self.n * dm                              # (H 1 N)
        a = F.softplus(self.a_param)[:, None]
        m_a = m / a                                  # (H A N)

        X_m = self.lct(x, m)
        psi_m_a = torch.exp(1j * pi * self.alpha * m_a**2) * torch.exp(-1/2 * self.sigma**2 * (m_a - self.f0)**2)

        exp_tmp = torch.exp(-1j * pi * self.alpha * m**2 * (1 + 1/a**2))
        G_m = X_m * psi_m_a.conj() * exp_tmp
        g_tmp = fft.ifft(fft.ifftshift(G_m))
        g_tmp = fft.fftshift(g_tmp)
        Wt = 1/torch.sqrt(a) * torch.exp(-1j*pi * self.gamma * self.n**2) * g_tmp


        if plot:
            Wt = rearrange(Wt, '... a n -> ... n a')
            Wt_plot = Wt[0, 2, :, :].cpu().numpy()
            X, Y = torch.meshgrid(self.n.float(), a[:, 0].float(), indexing="ij")
            X = X.cpu().numpy()
            Y = Y.cpu().numpy()
            plt.figure()
            plt.pcolormesh(X, Y, abs(Wt_plot), shading='auto', cmap='viridis')
            plt.colorbar(label="Value")
            plt.xlabel("n")
            plt.ylabel("a")
            plt.show()
            return None


        Wt_abs = torch.abs(Wt)
        Wt_abs = (Wt_abs - Wt_abs.mean(dim=(2, 3), keepdim=True)) / (Wt_abs.std(dim=(2, 3), keepdim=True) + 1e-8)
        logits = self.resnet(Wt_abs)
        return logits


