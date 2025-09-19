import json
import os
import matplotlib.pyplot as plt
import matplotlib
# print(matplotlib.matplotlib_fname()) #获取字体目录
# print(matplotlib.get_cachedir()) #清理缓存

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22

path1 = '/media/user/新加卷/CIFAR10_result/data_proj_clc0.1_cifar10/10000/metric-fid50k_full.jsonl'
# path1 = '/media/user/新加卷/CIFAR10_result/output_proj_cont_wgan_k1_0.05_k2_0.5_k3_0.01/10000/metric-fid50k_full.jsonl'
path2 = '/media/user/新加卷/CIFAR10_result/data_proj_clc0.1+cont0.5,0.01_constant_cifar10/10000/metric-fid50k_full.jsonl'
path3 = '/media/user/新加卷/CIFAR10_result/output_proj_cont_wgan_k1_0.1to0.05_constant_k2_0.5_k3_0.01/10000/metric-fid50k_full.jsonl'
# path2 = '/media/user/新加卷/CIFAR10_result/output_proj_cont_wgan_k1_0.075_k2_0.5_k3_0.01/10000/metric-fid50k_full.jsonl'
# path5 = '/media/user/新加卷/result/output_proj_wgan_k1_0.1to0.05/10000/metric-fid50k_full.jsonl'
path4 = '/media/user/新加卷/CIFAR10_result/output_proj_cont_wgan_k1_0.1to0.05_s8e4_k2_0.42to0.5_k3_0.008to0.01/10000_4090/metric-fid50k_full.jsonl'
# path4 = '/media/user/新加卷/result/output_proj_cont_wgan_k1_0.1to0.05_k2_0.5to0_k3_0.01to0/10000/metric-fid50k_full.jsonl'
# path5 = '/media/user/新加卷/result/output_proj_cont_wgan_k1_0.1to0.05_s8e4_e1.6e5_k2_0.5to0_k3_0.01to0/10000/metric-fid50k_full.jsonl'
# path6 = '/media/user/新加卷/result/output_proj_cont_wgan_k1_0.1to0.073_s8e4_e1.6e5_k2_0.5to0_k3_0.01to0/10000/metric-fid50k_full.jsonl'
# path7 = '/media/user/新加卷/result/output_proj_cont_wgan_k1_0.1to0.05_k2_0.5_781_1e5_k3_0.01/10000/metric-fid50k_full.jsonl'
# path8 = '/media/user/新加卷/result/output_proj_cont_wgan_k1_0.05_k2_0.5_k3_0.01/10000/metric-fid50k_full.jsonl'
# path9 = '/media/user/新加卷/result/output_proj_cont_wgan_k1_0.1to0.05_s8e4_e1.5e5_k2_0.5to0_k3_0.01/10000/metric-fid50k_full.jsonl'
# path11 = '/media/user/新加卷/result/proj_cont/10000/metric-fid50k_full.jsonl'
# path12 = '/media/user/新加卷/result/output_proj_cont_wgan_k1_0.1to0.05_k2_0.5to0.45_k3_0.01/10000/metric-fid50k_full.jsonl'
# path13 = '/media/user/新加卷/result/output_proj_cont_wgan_k1_0.1to0.05_k2_0.5to0.495_k3_0.01/10000/metric-fid50k_full.jsonl'
# path14 = '/media/user/新加卷/result/output_proj_cont_wgan_k1_0.1to0.05_s8e4_k2_0.5+_781_1e5_k3_0.01/10000/metric-fid50k_full.jsonl'
# path15 = '/media/user/新加卷/result/output_proj_cont_wgan_k1_0.1to0.05_k2_1to0.5_k3_0.02to0.01/10000/metric-fid50k_full.jsonl'
all_path = [path1, path2, path3, path4] #, path5, path6, path9, path12, path7, path13, path14, path15]
index = 0
dict_data = {}
for path in all_path:
    with open(path, 'r') as f:
        x = []
        y = []
        i = 0
        for line in f:
            i += 1
            data = json.loads(line)
            x.append(i * 10000)
            y.append(data['results']['fid50k_full'])
    dict_data[index] = [x, y]
    index += 1
plt.figure(figsize=(8, 7))
#5 4 7 9
# plt.plot(dict_data[0][0][:], dict_data[0][1][:], label=r'0.1_$R_D$+$R_G$', linewidth=3)
# plt.plot(dict_data[0][0][5], dict_data[0][1][5], '*', color='#1f77b4', alpha=1, markersize=22, zorder=15)
# plt.plot(dict_data[1][0][:], dict_data[1][1][:], label=r'0.15_$R_D$+$R_G$', linewidth=3)
# plt.plot(dict_data[1][0][4], dict_data[1][1][4], '*', color='#ff7f0e', alpha=1, markersize=22, zorder=15)
# plt.plot(dict_data[2][0][:10], dict_data[2][1][:10], label=r'0.2_$R_D$+$R_G$', linewidth=3)
# plt.plot(dict_data[2][0][7], dict_data[2][1][7], '*', color='#2ca02c', alpha=1, markersize=22, zorder=15)
# plt.plot(dict_data[3][0][:10], dict_data[3][1][:10], label=r'WD-$R_D$+$R_G$', linewidth=3)
# plt.plot(dict_data[3][0][9], dict_data[3][1][9], '*', color='#d62728', alpha=1, markersize=22, zorder=15, label='Minimum Value')
plt.plot(dict_data[0][0][:], [0]*13, label=r'EM', linewidth=3)
# # 4, 7, 9, 8
plt.plot(dict_data[0][0][:], dict_data[0][1][:], label=r'+$R_D$', linewidth=3)
plt.plot(dict_data[0][0][4], dict_data[0][1][4], '*', color='#ff7f0e', alpha=1, markersize=22, zorder=15)
plt.plot(dict_data[1][0][:], dict_data[1][1][:], label=r'+$R_D$+$R_G$', linewidth=3)
plt.plot(dict_data[1][0][7], dict_data[1][1][7], '*', color='#2ca02c', alpha=1, markersize=22, zorder=15)
plt.plot(dict_data[2][0][:13], dict_data[2][1][:13], label=r'+WD-$R_D$+$R_G$', linewidth=3)
plt.plot(dict_data[2][0][9], dict_data[2][1][9], '*', color='#d62728', alpha=1, markersize=22, zorder=15)
plt.plot(dict_data[3][0][:13], dict_data[3][1][:13], label=r'+WD-$R_D$+WA-$R_G$', linewidth=3)
plt.plot(dict_data[3][0][8], dict_data[3][1][8], '*', color='#9467bd', alpha=1, markersize=22, zorder=15, label='Minimum Value')
# plt.plot(dict_data[4][0][:], dict_data[4][1][:], label='RCLC')
# plt.plot(dict_data[4][0][:], dict_data[4][1][:], label='output_proj_cont_wgan_k1_0.1to0.05_s8e4_e1.6e5_k2_0.5to0_k3_0.01to0')
# plt.plot(dict_data[5][0][:], dict_data[5][1][:], label='output_proj_cont_wgan_k1_0.1to0.073_s8e4_e1.6e5_k2_0.5to0_k3_0.01to0')
# plt.plot(dict_data[6][0][:], dict_data[6][1][:], label='k1_0.1to0.05')
# plt.plot(dict_data[7][0][:], dict_data[7][1][:], label='output_proj_cont_wgan_k1_0.05_k2_0.5_k3_0.01')
# plt.plot(dict_data[8][0][:], dict_data[8][1][:], label='output_proj_cont_wgan_k1_0.1to0.05_s8e4_e1.5e5_k2_0.5to0_k3_0.01')
# plt.plot(dict_data[9][0][:], dict_data[9][1][:], label='output_proj_cont')
# plt.plot(dict_data[6][0][:], dict_data[6][1][:], label='output_proj_cont_wgan_k1_0.1to0.05_s8e4_e1.5e5_k2_0.5to0_k3_0.01')
# plt.plot(dict_data[7][0][:], dict_data[7][1][:], label='output_proj_cont_wgan_k1_0.1to0.05_k2_0.5to0.45_k3_0.01')
# plt.plot(dict_data[8][0][:], dict_data[8][1][:], label='output_proj_cont_wgan_k1_0.1to0.05_k2_0.5_781_1e5_k3_0.01')
# plt.plot(dict_data[9][0][:], dict_data[9][1][:], label='output_proj_cont_wgan_k1_0.1to0.05_k2_0.5to0.495_k3_0.01')
# plt.plot(dict_data[10][0][:], dict_data[10][1][:], label='output_proj_cont_wgan_k1_0.1to0.05_s8e4_k2_0.5+_781_1e5_k3_0.01')
# plt.plot(dict_data[11][0][:], dict_data[11][1][:], label='output_proj_cont_wgan_k1_0.1to0.05_k2_1to0.5_k3_0.02to0.01')
plt.legend(framealpha=1, fontsize=18)
plt.xlabel('Training Steps', fontsize=25)
plt.ylabel('FID50k', fontsize=25)# , rotation=0, labelpad=40
plt.title('CIFAR10 Dataset', fontsize=25)
plt.tight_layout()
plt.show()

