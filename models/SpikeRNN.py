import torch
from torch import nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


class SPE(nn.Module):
    def __init__(self, input_len, patch_num, patch_dim, T, tau, D):
        super().__init__()
        self.patch_projector = nn.Linear(input_len // patch_num, patch_dim)
        self.bn = nn.BatchNorm2d(patch_dim)
        self.encoder_lif = MultiStepLIFNode(tau=tau, detach_reset=False, backend='torch')

        self.D = D
        self.T = T
        self.patch_dim = patch_dim
        self.patch_num = patch_num

    def forward(self, x):
        B, L, D = x.shape

        x = x.view(B, self.patch_num, L // self.patch_num, D).contiguous()
        x = x.transpose(-1, -2).contiguous()
        x = self.patch_projector(x)
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.flatten(0, 1)
        x = self.bn(x)
        x = x.view(self.T, B, self.patch_dim, self.patch_num, D)
        x = self.encoder_lif(x)

        return x


# BUG FIX: removed unused `mean`, `last`, `std` constructor parameters
class SpikeRNN(nn.Module):
    def __init__(self, input_len, patch_num, patch_dim, T, blocks, D, pred_len, tau, alpha, hidden_dim):
        super().__init__()
        self.emb = SPE(input_len, patch_num, patch_dim, T, tau, D)

        self.recurrent = nn.ModuleList()
        self.rec_bns = nn.ModuleList()
        self.rec_lifs = nn.ModuleList()

        for i in range(blocks):
            self.recurrent.append(nn.Linear(patch_num + hidden_dim, hidden_dim))
            self.rec_bns.append(nn.BatchNorm2d(patch_dim))
            self.rec_lifs.append(MultiStepLIFNode(tau=tau, detach_reset=False, backend='torch'))

        self.dense1 = nn.Linear(patch_dim * (patch_num + hidden_dim), hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, pred_len)
        self.bn = nn.BatchNorm1d(D)
        self.activ = MultiStepLIFNode(tau=tau, detach_reset=False, backend='torch')
        self.hidden = hidden_dim

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        x = x - mean

        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std

        x = self.emb(x)
        T, B, pd, pn, D = x.shape
        h = torch.zeros([T, B, pd, self.hidden, D]).to(x.device)
        x0 = x.transpose(-1, -2)
        x = torch.cat((x, h), dim=-2).contiguous()
        x = x.transpose(-1, -2).contiguous()

        for i in range(len(self.recurrent)):
            h = self.recurrent[i](x)
            x = torch.cat((x0, h), dim=-1).contiguous()
            x = x.flatten(0, 1)
            x = self.rec_bns[i](x)
            x = x.view(T, B, pd, D, -1)
            x = self.rec_lifs[i](x)

        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.flatten(-2, -1)
        x = self.dense1(x)
        x = x.flatten(0, 1)
        x = self.bn(x)
        x = x.view(T, B, D, -1)
        x = self.activ(x)
        x = self.dense2(x)
        x = x.transpose(-1, -2).contiguous()
        x = x.view(T, B, -1, D)

        x = x * std
        x = x + mean.repeat(T, 1, 1, 1)

        return x.mean(dim=0)
