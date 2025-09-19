import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
##k[0 0 0] and k[0.5 0.5 0.5]##
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'cm'
# rcParams['text.usetex'] = True
# rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

x = np.linspace(-4, 4, 9)
y = np.linspace(-4, 4, 9)
x_loc = np.tile(x, 9)
y_loc = np.repeat(y, 9)
m = np.array([0.5, 1, -1, 0.8])
phi = np.zeros((4, 10000))
theta = np.zeros(10000)
phi[:, 0] = 2
theta[0] = 2
lr = 0.1
k1 = 0.5
k2 = 0.5
k3 = 0.5
for i in range(9999):
    # phi_1[i + 1] = phi_1[i] - lr * (-m[0] * theta[i] + k1 * m[0] ** 2 * phi_1[i])
    # phi_2[i + 1] = phi_2[i] - lr * (-m[1] * theta[i] + k1 * m[1] ** 2 * phi_2[i])
    # phi_3[i + 1] = phi_3[i] - lr * (-m[2] * theta[i] + k1 * m[2] ** 2 * phi_3[i])
    # phi_4[i + 1] = phi_4[i] - lr * (-m[3] * theta[i] + k1 * m[3] ** 2 * phi_4[i])
    phi[:, i+1] = phi[:, i] - lr * (m * theta[i] + k1 * m ** 2 * phi[:, i])
    theta[i+1] = theta[i] - lr * (sum(-m * phi[:, i+1] + k3 * m**2 * theta[i]) + k2 * theta[i])
fig, axs = plt.subplots(1, 4)
axs[0].scatter(theta, phi[0, :], s=3, c='blue', marker='^')
axs[0].scatter(2, 2, s=15, c='red', marker='o')
color = [(255/255, 128/255, 0/255)]
axs[0].scatter(theta[-1], phi[0, -1], s=15, c=color, marker='o')
axs[0].scatter(0, 0, s=10, c='green', marker='*')
axs[0].set_xlabel(r'$\hat{\theta}$', fontweight='bold')
axs[0].set_ylabel(r'$\phi_1$', fontweight='bold', rotation=0)
axs[0].set_aspect('equal')
axs[0].set_xlim([-4, 4])
axs[0].set_ylim([-4, 4])
for i in y:
    for j in x:
        if i == 0 and j == 0:
            continue
        dx = -m[0] * i - k3 * m[0] ** 2 * j - k2 * j
        dy = m[0] * j - k1 * m[0] ** 2 * i
        axs[0].arrow(j, i, 0.2 * dx, 0.2 * dy, head_width=0.05, head_length=0.1, fc='k', ec='k')

axs[1].scatter(theta, phi[1, :], s=3, c='blue', marker='^')
axs[1].scatter(2, 2, s=15, c='red', marker='o')
axs[1].scatter(theta[-1], phi[1, -1], s=15, c=color, marker='o')
axs[1].scatter(0, 0, s=10, c='green', marker='*')
axs[1].set_xlabel(r'$\hat{\theta}$')
axs[1].set_ylabel(r'$\phi_2$', rotation=0)
axs[1].set_aspect('equal')
axs[1].set_xlim([-4, 4])
axs[1].set_ylim([-4, 4])
for i in y:
    for j in x:
        if i == 0 and j == 0:
            continue
        dx = -m[1] * i - k3 * m[1] ** 2 * j - k2 * j
        dy = m[1] * j - k1 * m[1] ** 2 * i
        axs[1].arrow(j, i, 0.2 * dx, 0.2 * dy, head_width=0.05, head_length=0.1, fc='k', ec='k')

axs[2].scatter(theta, phi[2, :], s=3, c='blue', marker='^')
axs[2].scatter(2, 2, s=15, c='red', marker='o')
axs[2].scatter(theta[-1], phi[2, -1], s=15, c=color, marker='o')
axs[2].scatter(0, 0, s=10, c='green', marker='*')
axs[2].set_xlabel(r'$\hat{\theta}$')
axs[2].set_ylabel(r'$\phi_3$', rotation=0)
axs[2].set_aspect('equal')
axs[2].set_xlim([-4, 4])
axs[2].set_ylim([-4, 4])
for i in y:
    for j in x:
        if i == 0 and j == 0:
            continue
        dx = -m[2] * i - k3 * m[2] ** 2 * j - k2 * j
        dy = m[2] * j - k1 * m[2] ** 2 * i
        axs[2].arrow(j, i, 0.2 * dx, 0.2 * dy, head_width=0.05, head_length=0.1, fc='k', ec='k')

axs[3].scatter(theta, phi[3, :], s=3, c='blue', marker='^')
axs[3].scatter(2, 2, s=15, c='red', marker='o')
axs[3].scatter(theta[-1], phi[3, -1], s=15, c=color, marker='o')
axs[3].scatter(0, 0, s=10, c='green', marker='*')
axs[3].set_xlabel(r'$\hat{\theta}$')
axs[3].set_ylabel(r'$\phi_4$', rotation=0)
axs[3].set_aspect('equal')
axs[3].set_xlim([-4, 4])
axs[3].set_ylim([-4, 4])
for i in y:
    for j in x:
        if i == 0 and j == 0:
            continue
        dx = -m[3] * i - k3 * m[3] ** 2 * j - k2 * j
        dy = m[3] * j - k1 * m[3] ** 2 * i
        axs[3].arrow(j, i, 0.2 * dx, 0.2 * dy, head_width=0.05, head_length=0.1, fc='k', ec='k')

plt.tight_layout()
plt.savefig("dirac_projected_wgan.png", bbox_inches="tight", pad_inches=0.)
plt.show()