import random

import torch
import numpy as np
import torch.nn as nn
from random_queue import Random_queue
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import math
from gen_color import gen_color_map
from torch import autograd
# dataset
class GridSampler(object):
    def __init__(self, n_row=4, n_col=None, edge=1.0, sigma=0.02, n_per_mode=50):
        self.n_row = n_row
        self.n_col = n_col if n_col else n_row
        self.edge = edge
        self.sigma = sigma
        self.n_per_mode = n_per_mode

        self.centers = None
        self.n_mode = None
        self.n_data = None

        self.build()

    def build(self):
        x = np.linspace(-4, 4, self.n_row) * self.edge
        y = np.linspace(-4, 4, self.n_col) * self.edge
        X, Y = np.meshgrid(x, y)

        self.centers = np.stack((X.flatten(), Y.flatten()), 1)  #shape n_row , 2
        self.n_mode = self.n_row * self.n_col

        self.n_data = self.n_mode * self.n_per_mode
        self.data = np.repeat(self.centers, self.n_per_mode, axis=0) + np.random.normal(size=(self.n_data, 2)) * self.sigma # shape n_per_mode * n_row, 2
        # self.idx = np.random.choice(self.n_mode, self.n_data, p=[0.76, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        # self.data = self.centers[self.idx] + np.random.normal(size=(self.n_data, 2)) * self.sigma

        # self.labels = np.random.randint(0, self.n_class, size=(self.n_data,), dtype=np.int64)

    @property
    def x_range(self):
        return (-6, 6)

    @property
    def y_range(self):
        return (-6, 6)

class RingSampler(object):
    def __init__(self, n_mode=12, radius=2, sigma=0.1, n_per_mode=50):
        self.n_mode = n_mode
        self.radius = radius
        self.sigma = sigma
        self.n_per_mode = n_per_mode

        self.centers = None
        self.n_data = None

        self.build()

    def build(self):
        self.centers = np.stack([[self.radius * np.cos(2 * np.pi / self.n_mode * i),
                                 self.radius * np.sin(2 * np.pi / self.n_mode * i)] for i in range(self.n_mode)], 0)  #shape n_row , 2

        self.n_data = self.n_mode * self.n_per_mode
        self.data = np.repeat(self.centers, self.n_per_mode, axis=0) + np.random.normal(size=(self.n_data, 2)) * self.sigma # shape n_per_mode * n_row, 2
        # self.idx = np.random.choice(self.n_mode, self.n_data, p=[0.76, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        # self.data = self.centers[self.idx] + np.random.normal(size=(self.n_data, 2)) * self.sigma

        # self.labels = np.random.randint(0, self.n_class, size=(self.n_data,), dtype=np.int64)

    @property
    def x_range(self):
        return (-2.5, 2.5)

    @property
    def y_range(self):
        return (-2.5, 2.5)

class XSampler(object):
    def __init__(self, data_type, use_cuda=False, n_class=None, **kwargs):
        self.data_type = data_type

        n_per_mode = kwargs.get("n_per_mode", 100)

        if data_type == "grid":
            n_row = kwargs.get("n_grid", 5)
            n_col = n_row
            sampler = GridSampler(n_row=n_row, n_col=n_col, edge=1.0, sigma=0.1, n_per_mode=n_per_mode)

        elif data_type == "ring":
            n_mode = kwargs.get("n_ring", 12)
            sampler = RingSampler(n_mode=n_mode, radius=2, sigma=0.1, n_per_mode=n_per_mode)
        #
        # elif data_type == "circle":
        #     p_center = kwargs.get("p_center", 0.03)
        #     sampler = CircleSampler(p_center=p_center, radius=1.0, sigma=0.03)

        else:
            raise ValueError("%s data not supported." % data_type)

        self.sampler = sampler
        self.n_data = self.sampler.n_data
        self.use_cuda = use_cuda

        if n_class is not None:
            # self.labels = np.random.randint(0, n_class, size=(self.n_data,), dtype=np.int64)
            self.labels = np.repeat(range(n_class), repeats=n_per_mode, axis=0)
        else:
            self.labels = np.zeros(shape=(self.n_data,), dtype=np.int64)

        self.n_class = np.max(self.labels) + 1

    def __len__(self):
        return self.n_data

    def special_data_iter(self, idx: list, batch_size: int = 128, shuffle: bool = True, drop_last: bool = True,
                          onehot: bool = True):

        if shuffle:
            np.random.shuffle(idx)

        n_data = len(idx)

        s = 0

        while s < n_data:

            e = s + batch_size

            if drop_last:
                if e > n_data:
                    return
            else:
                if e > n_data:
                    e = n_data

            ii = idx[s: e]
            data = torch.from_numpy(self.sampler.data[ii]).float()

            if onehot:
                label = torch.from_numpy(np.eye(self.n_class)[self.labels[ii]]).float()
            else:
                label = torch.from_numpy(np.array(self.labels[ii])).long()

            if self.use_cuda:
                data = data.cuda()
                label = label.cuda()

            yield data, label, ii

            s = e

        return

    def data_iter(self, batch_size=128, shuffle=True, drop_last=True, onehot=True):
        idx = np.arange(self.n_data)

        return self.special_data_iter(idx, batch_size, shuffle, drop_last=drop_last, onehot=onehot)

    @property
    def x_range(self):
        return self.sampler.x_range

    @property
    def y_range(self):
        return self.sampler.y_range

    def sample(self, n_samples=100):
        """
        sample.
        """
        n_samples = min(n_samples, self.n_data)
        idx = np.random.permutation(np.arange(self.n_data))[:n_samples]

        data = self.sampler.data[idx]
        label = self.labels[idx]

        return data, label, idx

    def sample_class(self, idx, num):
        """
        Sample specific class of data.
        """
        target_idx = np.where(self.labels == idx)[0]
        np.random.shuffle(target_idx)

        data = self.sampler.data[target_idx[:num]]
        data = torch.from_numpy(data).float()

        if self.use_cuda:
            data = data.cuda()

        return data

    def __getitem__(self, idx):
        if isinstance(idx, int) or isinstance(idx, slice) or isinstance(idx, np.ndarray):
            return self.sampler.data[idx]

        else:
            raise ValueError("invalid input: {}, should be either int, slice, or np.ndarray".format(idx))

# model
class Generator(nn.Module):
    def __init__(self, zc_dim=32, n_hidden=128, in_channels=2):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(zc_dim, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, in_channels)
        # self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()

    def forward(self, x, output_f=False):
        x = self.activation(self.linear1(x))
        f = self.activation(self.linear2(x))
        x = self.linear3(f)

        if output_f:
            return x, f

        return x


class Discriminator(nn.Module):
    def __init__(self, n_hidden=128, x_depth=2, ifsigmoid=True):
        super(Discriminator, self).__init__()

        self.ifsigmoid = ifsigmoid

        self.linear1 = nn.Linear(x_depth, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        # self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        if self.ifsigmoid:
            x = torch.sigmoid(x)

        return x


class CDiscriminator(nn.Module):
    def __init__(self, n_hidden=128, x_depth=2, c_depth=1):
        super(CDiscriminator, self).__init__()

        self.linear1 = nn.Linear(x_depth + c_depth, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        # self.activation = nn.ReLU()

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        feat = x
        # x = torch.sigmoid(self.linear3(x))
        x = self.linear3(x)

        return x, feat


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if (type(m) == nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.zeros_(m.bias.data)


def visualize_pts2(real, pts, idx, n_class, xrange=(-3.0, 3.0), yrange=(-3.0, 3.0), savename="pts.png", weights=None, title="points", color=None):
    """
    Args:
        pts: np.ndarray, shape in `[N, 2]`.
        idx: list or 1-dim np.ndarray, shape in `[N]`.
        n_class: int, number of total classes.
        xrange: tuple, (xmin, xmax).
        yrange: tuple, (ymin, ymax).
        savename: str, name of saved file.
        title: title of the whole figure.
    """

    fig, ax = plt.subplots()

    # for i in range(n_class):
    #     i_idx = np.where(idx == i)[0]
        # ax.scatter(pts[i_idx, 0], pts[i_idx, 1], alpha=0.5, label="%d" % i)
    ax.plot(pts[:, 0], pts[:, 1], ".", alpha=1, label="%d" % 0, c=color['0'])
    syn_data = real.sampler.data
    ax.plot(syn_data[:, 0], syn_data[:, 1], '.', alpha=0.05, label='1', c='red')

    # ax.legend(ncol=math.ceil(n_class / 10), bbox_to_anchor=(1.0, 1.0))
    if xrange is not None and yrange is not None:
        ax.set_xlim(xmin=xrange[0], xmax=xrange[1])
        ax.set_ylim(ymin=yrange[0], ymax=yrange[1])

    if title is not None:
        ax.set_title(title, fontsize=30)
    ax.set_xticks(fontsize=20)
    ax.set_yticks(fontsize=20)

    plt.tight_layout()
    # plt.legend()
    fig.savefig(savename)
    plt.clf(); plt.cla(); plt.close()

def compute_grad2(d_out, x_in): # 平方
    batch_size = d_out.size(0)
    reg = []
    grad_dout = autograd.grad(outputs=d_out.sum(),
                              inputs=x_in,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg.append(grad_dout2.view(batch_size, -1).sum(1))
    return reg

def wgan_gp_reg(D, x_real, x_fake, y, center=1.):
    batch_size = y.size(0)
    eps = torch.rand(batch_size, device=y.device).view(batch_size, 1)
    x_interp = (1 - eps) * x_real + eps * x_fake
    x_interp = x_interp.detach()
    x_interp.requires_grad_()
    d_out, _ = D(x_interp, y)

    reg = sum([(i.sqrt() - center).pow(2).mean() for i in compute_grad2(d_out, x_interp)])

    return reg

# main
use_cuda = torch.cuda.is_available()


def setup(x):
    if use_cuda:
        x = x.cuda()

    return x

def main():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    batch_size = 64
    batch = 0
    epoch = 0
    n_epoch = 30000
    z_dim = 10
    c_dim = 1
    MSE = nn.MSELoss()
    data_dict = {"n_grid": 5, "n_ring": 12, "n_per_mode": 100, "n_class": None}

    x_sample = XSampler(data_type='grid', use_cuda=use_cuda, **data_dict)

    color_dict = gen_color_map(c_dim)

    zc_dim = z_dim + c_dim
    hidden_dim = 64

    G = Generator(zc_dim=zc_dim, n_hidden=hidden_dim, in_channels=2)
    # G.load_state_dict(torch.load('./visual/wgan_clc0.1_mse0.1_1class_circle/G.pth'))
    setup(G)
    G.apply(weights_init)

    D = CDiscriminator(n_hidden=hidden_dim, x_depth=2, c_depth=1)
    # D.load_state_dict(torch.load('./visual/wgan_clc0.1_mse0.1_1class_circle/D.pth'))
    setup(D)
    D.apply(weights_init)

    G_Optimizer = torch.optim.RMSprop(G.parameters(), lr=1e-3, alpha=0.99, eps=1e-8)
    D_Optimizer = torch.optim.RMSprop(D.parameters(), lr=1e-3, alpha=0.99, eps=1e-8)

    z_sampler = torch.distributions.Normal(0, 1)

    logger = SummaryWriter('./monitoring')

    ztest = torch.distributions.Normal(0, 1)

    logits = torch.zeros(1, device='cuda')
    ytest = torch.distributions.categorical.Categorical(logits=logits)

    # 训练
    G.train()
    D.train()
    f_queue = Random_queue(capacity=6400, batch_size=64)
    r_queue = Random_queue(capacity=6400, batch_size=64)
    dloss_buffer = deque(maxlen=2)
    gloss_buffer = deque(maxlen=2)

    # 初次可视化
    with torch.no_grad():
        z_test = ztest.sample((2500, z_dim))
        z_test = setup(z_test)
        idx = ytest.sample((2500, 1))
        # y_test = torch.eye(25, device='cuda')[idx]
        # y_test = setup(y_test)
        pts = G(torch.cat([z_test, idx], dim=1))
        visualize_pts2(real=x_sample, pts=pts.cpu().numpy(), idx=idx.cpu().numpy(), n_class=1, savename='./visual/generation_init.png', xrange=x_sample.x_range, yrange=x_sample.y_range,
                      title="Step 0", color={'0':'blue'})

    while epoch < n_epoch:
        data_iter = x_sample.data_iter(batch_size=batch_size, shuffle=True, onehot=False)
        epoch += 1
        for data, label, ii in data_iter:
            batch += 1
            D_Optimizer.zero_grad()
            z = z_sampler.sample((batch_size, z_dim))
            z = setup(z)
            label = label.unsqueeze(-1)
            with torch.no_grad():
                gen = G(torch.cat([z, label], dim=1))
            d_fake, _ = D(gen, label)
            d_real, _ = D(data, label)

            xtmp = gen.detach().cpu().numpy()
            ytmp = label.detach().cpu().numpy()
            f_queue.set_data(xtmp, ytmp)

            xtmp = data.detach().cpu().numpy()
            r_queue.set_data(xtmp, ytmp)

            f_pre, label_pre = f_queue.get_data(batch_size=64)
            r_pre, label_pre = r_queue.get_data(batch_size=64)
            f_pre = torch.as_tensor(f_pre, dtype=torch.float32).cuda()
            r_pre = torch.as_tensor(r_pre, dtype=torch.float32).cuda()
            label_pre = torch.as_tensor(label_pre, dtype=torch.long).cuda()

            di_f, _ = D(f_pre, label_pre)
            di_r, _ = D(r_pre, label_pre)

            k_1 = max(1 * (1 - batch / 1.2e6), 0.5)
            i_loss = (di_f ** 2 + di_r ** 2).mean() * k_1
            d_w = (d_real - d_fake).mean()
            d_loss = d_w + i_loss
            # gp = wgan_gp_reg(D, data, gen, label, center=0) * 0.1
            # d_w = (d_real - d_fake).mean()
            # d_loss = d_w + gp

            dloss_buffer.append(d_loss.item())
            d_loss.backward()
            D_Optimizer.step()

            G_Optimizer.zero_grad()
            z = z_sampler.sample((batch_size, z_dim))
            z = setup(z)
            gen = G(torch.cat([z, label], dim=1))
            d_fake, _ = D(gen, label)
            k_2 = 0.01#0.001 - 1/ 1.2e10 * batch#0.001 - 1 / 1.2e10 * batch#max(0.005 * (1 - batch / 1.5e6), 0.001)#0.1 if batch < 3e5 else max(0.1 * (1 - (batch - 3e5) / 1e6), 0)
            l2_reg = MSE(gen, data) * k_2
            g_w = d_fake.mean()
            g_loss = g_w + l2_reg

            gloss_buffer.append(g_loss.item())
            g_loss.backward()
            G_Optimizer.step()

            if (batch % 1000) == 0:
                logger.add_scalar('{}/{}'.format('D', 'd_w'), d_w.item(), batch)
                logger.add_scalar('{}/{}'.format('D', 'i_loss'), i_loss.item(), batch)
                logger.add_scalar('{}/{}'.format('D', 'loss'), d_loss.item(), batch)

                logger.add_scalar('{}/{}'.format('G', 'g_w'), g_w.item(), batch)
                logger.add_scalar('{}/{}'.format('G', 'l2_reg'), l2_reg.item(), batch)
                logger.add_scalar('{}/{}'.format('G', 'loss'), g_loss.item(), batch)

                print(f'{epoch}_{batch} done dloss:{dloss_buffer[-1]:04f} gloss:{gloss_buffer[-1]:04f}')

        if (epoch % 500) == 0:

            with torch.no_grad():
                z_test = ztest.sample((2500, z_dim))
                z_test = setup(z_test)
                idx = ytest.sample((2500, 1))
                # y_test = torch.eye(25, device='cuda')[idx]
                # y_test = setup(y_test)
                pts = G(torch.cat([z_test, idx], dim=1))
                visualize_pts2(real=x_sample, pts=pts.cpu().numpy(), idx=idx.cpu().numpy(), n_class=1,
                               savename='./visual/generation_{}.png'.format(epoch), xrange=x_sample.x_range,
                               yrange=x_sample.y_range,
                               title="Step {}".format(epoch),
                               color={'0':'blue'})
            torch.save(G.state_dict(), './G.pth')
            torch.save(D.state_dict(), './D.pth')


if __name__ == '__main__':
    main()







