from main import Generator, visualize_pts2, XSampler
import torch
import matplotlib
# matplotlib.use('svg')
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'cm'

use_cuda = torch.cuda.is_available()
def setup(x):
    if use_cuda:
        x = x.cuda()

    return x

z_dim = 10
c_dim = 1
zc_dim = z_dim + c_dim
hidden_dim = 64
G1 = Generator(zc_dim=zc_dim, n_hidden=hidden_dim, in_channels=2)
G1.load_state_dict(torch.load('/home/user/Projects/SynthesisGAN/visual/wgan_clcl1_1class_grid/2/G.pth'))
G1 = setup(G1)

G2 = Generator(zc_dim=zc_dim, n_hidden=hidden_dim, in_channels=2)
G2.load_state_dict(torch.load('/home/user/Projects/SynthesisGAN/visual/wgan_clc0.1_mse0.01_1class_grid/3_k1_1_k2_0.001/G.pth'))
G2 = setup(G2)

G3 = Generator(zc_dim=zc_dim, n_hidden=hidden_dim, in_channels=2)
G3.load_state_dict(torch.load('/home/user/Projects/SynthesisGAN/visual/wgan_clc0.1_mse0.01to0_1class_grid/3_clc1_mse0.001to0.0009/G.pth'))
G3 = setup(G3)

data_dict = {"n_grid": 5, "n_ring": 12, "n_per_mode": 100, "n_class": None}

x_sample = XSampler(data_type='grid', use_cuda=use_cuda, **data_dict)

ztest = torch.distributions.Normal(0, 1)

logits = torch.zeros(1, device='cuda')
ytest = torch.distributions.categorical.Categorical(logits=logits)
with (torch.no_grad()):
    # for i in range(5):
    # plt.switch_backend('pdf')
    fig, ax = plt.subplots(1, 3, figsize=(8, 7))
    color = {'0': 'blue'}
    xrange = (-5, 5)#x_sample.x_range
    yrange = (-5, 5)#x_sample.y_range

    z_test = ztest.sample((2500, z_dim))
    z_test = setup(z_test)
    idx = ytest.sample((2500, 1))
    pts = G1(torch.cat([z_test, idx], dim=1))
    pts = pts.cpu().numpy()
        # visualize_pts2(real=x_sample, pts=pts.cpu().numpy(), idx=idx.cpu().numpy(), n_class=1, savename='/home/user/Projects/SynthesisGAN/visual/wgan_clc0.1_mse0.01_1class_grid/3_k1_1_k2_0.001/final3_{}.svg'.format(i), xrange=x_sample.x_range, yrange=x_sample.y_range,
        #               title="Step 30000", color={'0':'blue'})
    ax[0].plot(pts[:, 0], pts[:, 1], ".", alpha=1, label='Generated Samples', c=color['0'])
    syn_data = x_sample.sampler.data
    ax[0].plot(syn_data[:, 0], syn_data[:, 1], '.', alpha=0.04, label='Real Samples', c='red')
    ax[0].set_xlim(xmin=xrange[0], xmax=xrange[1])
    ax[0].set_ylim(ymin=yrange[0], ymax=yrange[1])
    ax[0].set_title(r'EM+$R_D$', fontsize=20)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].legend(loc='upper right', fontsize=15)  # 调用 legend 显示图例
    # ax[0].set_yticks(fontsize=20)

    z_test = ztest.sample((2500, z_dim))
    z_test = setup(z_test)
    idx = ytest.sample((2500, 1))
    pts = G2(torch.cat([z_test, idx], dim=1))
    pts = pts.cpu().numpy()
    ax[1].plot(pts[:, 0], pts[:, 1], ".", alpha=1, label='Generated Samples', c=color['0'])
    syn_data = x_sample.sampler.data
    ax[1].plot(syn_data[:, 0], syn_data[:, 1], '.', alpha=0.04, label='Real Samples', c='red')
    ax[1].set_xlim(xmin=xrange[0], xmax=xrange[1])
    ax[1].set_ylim(ymin=yrange[0], ymax=yrange[1])
    ax[1].set_title(r'EM+$R_D$+$R_G$', fontsize=20)
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].legend(loc='upper right', fontsize=15)  # 调用 legend 显示图例
    # ax[1].set_yticks(fontsize=20)

    z_test = ztest.sample((2500, z_dim))
    z_test = setup(z_test)
    idx = ytest.sample((2500, 1))
    pts = G3(torch.cat([z_test, idx], dim=1))
    pts = pts.cpu().numpy()
    ax[2].plot(pts[:, 0], pts[:, 1], ".", alpha=1, label='Generated Samples', c=color['0'])
    syn_data = x_sample.sampler.data
    ax[2].plot(syn_data[:, 0], syn_data[:, 1], '.', alpha=0.04, label='Real Samples', c='red')
    ax[2].set_xlim(xmin=xrange[0], xmax=xrange[1])
    ax[2].set_ylim(ymin=yrange[0], ymax=yrange[1])
    ax[2].set_title(r'EM+$R_D$+WA-$R_G$', fontsize=20)
    ax[2].tick_params(axis='both', labelsize=20)
    ax[2].legend(loc='upper right', fontsize=15)  # 调用 legend 显示图例
    # ax[2].set_yticks(fontsize=20)

    plt.tight_layout()
    # plt.legend()
    # plt.savefig('./final.png', bbox_inches='tight', pad_inches=0)
    # plt.clf(); plt.cla(); plt.close()
    # plt.savefig('output.eps', format='eps')
    plt.show()


