import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
from gan_training.models.projector import F_RandomProj


class Generator(nn.Module):
    def __init__(self,
                 z_dim,
                 nlabels,
                 size,
                 embed_size=256,
                 nfilter=64,
                 nfilter_max=512,
                 **kwargs):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.z_dim = z_dim

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers - i), nf_max)
            nf1 = min(nf * 2**(nlayers - i - 1), nf_max)
            blocks += [ResnetBlock(nf0, nf1), nn.Upsample(scale_factor=2)]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z, y):
        assert (z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding(y)
        else:
            yembed = y

        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)

        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)

        out = self.resnet(out)

        out = self.conv_img(actvn(out))
        out = torch.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self,
                 nlabels,
                 size,
                 in_channel,
                 nfilter=64,
                 nfilter_max=1024):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nlayers = nlayers
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(in_channel, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, nlabels) # 相当于分类器
        # self.dc = nn.Conv2d(3 * nf, nf, 1)

    def forward(self, x, y):
        assert (x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.conv_img(x)
        # if x_a is not None:
        #     out = self.dc(torch.cat([out, x_a], dim=1))
        # for i, layers in enumerate(self.resnet):
        #     out = layers(out)
        #     if i == 2: x_a = out
        out = self.resnet(out)

        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(actvn(out))

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        out = out[index, y]

        return out


class MultuDiscrim(nn.Module):
    def __init__(self, channels, resolutions, nlabels, num_disc=4, **kwargs):
        super().__init__()

        assert num_disc in [1, 2, 3, 4]

        self.disc_in_channels = channels[:num_disc]
        self.disc_in_res = resolutions[:num_disc]
        Disc = Discriminator

        multi_disc = dict()
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            multi_disc[str(i)] = Disc(nlabels=nlabels, size=res, in_channel=cin, **kwargs)

        self.multi_disc = nn.ModuleDict(multi_disc)

        # multi_mlp = dict()
        # for i in range(num_disc):
        #     multi_mlp[str(i)] = nn.Linear(in_features=num_disc, out_features=1)
        # self.multi_mlp = nn.ModuleDict(multi_mlp)
        # self.atten = SelfAttention(input_dim=nlabels)

    def forward(self, x, y):
        all_logits = []
        for k, disc in self.multi_disc.items():
            out = disc(x[k], y)
            # out = out.unsqueeze(-1)
            all_logits.append(out)

        return all_logits

        # fuse_matrix = torch.cat(all_logits, dim=-1)
        # fuse_out = []

        # for k, mlp in self.multi_mlp.items():
        #     out1 = mlp(fuse_matrix)
        #     out1 = out1.squeeze(-1) + all_logits[int(k)].squeeze(-1)
        #     index = Variable(torch.LongTensor(range(out1.size(0))))
        #     if y.is_cuda:
        #         index = index.cuda()
        #     out1 = out1[index, y]
        #     fuse_out.append(out1)
        # fuse_feature = self.atten(fuse_matrix)
        # for i in range(4):
        #     all_logits[i] = all_logits[i].squeeze(1)
        #     all_logits[i] = all_logits[i] + fuse_feature[:, i, :]
        #     index = Variable(torch.LongTensor(range(all_logits[i].size(0))))
        #     if y.is_cuda:
        #         index = index.cuda()
        #     all_logits[i] = all_logits[i][index, y]

        # return fuse_out


class ProjectedDisc(nn.Module):
    def __init__(self, nlabels, interp256=True, **kwargs):
        super().__init__()
        self.interp256 = interp256
        self.nlabels = nlabels
        self.feature_network = F_RandomProj(im_res=256, cout=64, expand=True, proj_type=2)
        self.multi_disc = MultuDiscrim(channels=self.feature_network.CHANNELS, resolutions=self.feature_network.RESOLUTIONS,
                                 nlabels=nlabels, **kwargs)

    def train(self, mode=True):
        self.feature_network =self.feature_network.train(False)
        self.multi_disc = self.multi_disc.train(mode)

    def eval(self):
        return self.train(False)

    def forward(self, x, y):
        if self.interp256:
            x = F.interpolate(x, 256, mode='bilinear', align_corners=False)

        feature = self.feature_network(x)
        all_logits = self.multi_disc(feature, y)

        return feature, all_logits


class Reconstructor(nn.Module):
    def __init__(self,
                 z_dim,
                 nlabels,
                 size,
                 embed_size=256,
                 nfilter=64,
                 nfilter_max=1024):
        super().__init__()
        self.z_dim = z_dim
        self.embed_size = embed_size
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, z_dim)

    def forward(self, x, y):
        assert (x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(actvn(out))

        # index = Variable(torch.LongTensor(range(out.size(0))))
        # if y.is_cuda:
        #     index = index.cuda()
        # out = out[index, y]

        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden,
                                self.fout,
                                3,
                                stride=1,
                                padding=1,
                                bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin,
                                    self.fout,
                                    1,
                                    stride=1,
                                    padding=0,
                                    bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.relu(x)
    # out = F.leaky_relu(x, 0.2)
    return out


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim, bias=False)
        self.key = nn.Linear(input_dim, input_dim, bias=False)
        self.value = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):
        batch_size = x.size(0) # 64
        # 查询、键、值的线性变换
        queries = self.query(x) # 64,4, 10
        keys = self.key(x)
        values = self.value(x)

        # 计算注意力权重
        scores = torch.matmul(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        scores = torch.softmax(scores, dim=2)

        # 注意力权重与值的点乘
        context = torch.matmul(scores, values)

        return context
