import torch
import torch.nn as nn


class Hellinger(nn.Module):
    def __init__(self, is_g):
        super().__init__()
        self.softmax_g = nn.Softmax(dim=2)
        self.is_g = is_g
        self.softmax_d = nn.Softmax(dim=0)

    def forward(self, gen, tgt):
        if self.is_g:
            N, C, H, W = gen.size()
            gen = gen.view(N, C, -1)
            tgt = tgt.view(N, C, -1)
            dist_gen = self.softmax_g(gen)
            dist_tgt = self.softmax_g(tgt)
            hellinger_loss = torch.sum((torch.sqrt(dist_gen) - torch.sqrt(dist_tgt)) ** 2)

            return hellinger_loss
        else:
            dist_gen = [self.softmax_d(D_g) for D_g in gen]
            dist_tgt = [self.softmax_d(D_r) for D_r in tgt]
            hellinger_loss = sum([torch.sum((torch.sqrt(d_g) - torch.sqrt(d_r)) ** 2) for d_g, d_r in zip(dist_gen, dist_tgt)])

            return hellinger_loss


# a = torch.rand((64, 3, 256, 256)).cuda()
# print(a.size())
# a = a / 0.5 - 1
# b = torch.rand((64, 3, 256, 256)).cuda()
# b = b / 0.5 - 1
# h = Hellinger()
# loss = h(a, b)
# print(loss)
