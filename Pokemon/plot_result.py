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


path_proj = '/media/user/新加卷/Pokemon_result/36_ProjectedGAN/metric-fid50k_full.jsonl'
path_ori = '/media/user/新加卷/Pokemon_result/data_ori/metric-fid50k_full.jsonl'
path_ori_cont = '/media/user/新加卷/Pokemon_result/data_proj_clc0.1_cont_2.5,0.01_constant/metric-fid50k_full.jsonl'
path_clc_reduce_cont_constant = '/media/user/新加卷/Pokemon_result/data_cont_constant_2.5_0.01/metric-fid50k_full.jsonl'
path_clc_cont_reduce = '/media/user/新加卷/Pokemon_result/data_cont(0.01-0)/metric-fid50k_full.jsonl'
all_path = [path_proj, path_ori, path_ori_cont, path_clc_reduce_cont_constant, path_clc_cont_reduce]
index = 0
dict_data = {}
for path in all_path:
    with open(path, 'r') as f:
        x = []
        y = []
        for line in f:
            data = json.loads(line)
            try :
                filename = os.path.splitext(os.path.basename(data['snapshot_pkl']))[0]
            except TypeError:
                print(data)
            x_cordinate = int(filename.split('_')[-1]) * 4e3 / 32
            x.append(x_cordinate)
            y.append(data['results']['fid50k_full'])
    dict_data[index] = [x, y]
    index += 1
# plt.plot(dict_data[0][0][10:], dict_data[0][1][10:], label='ORI')
minimum_x = [575 * 4e3 / 32, 2525 * 4e3 / 32, 800 * 4e3 / 32, 2300 * 4e3 / 32, 3150 * 4e3 / 32]
minimun_y = [27.73, 26.95, 26.95, 26.11, 25.95]
plt.figure(figsize=(8, 7))
plt.plot(dict_data[0][0][10:], dict_data[0][1][10:], label=r'Hinge Loss', linewidth=3, alpha=1, zorder=10)
plt.plot(minimum_x[0], minimun_y[0], '*', color='#1f77b4', alpha=1, markersize=22, zorder=15)
plt.plot(dict_data[1][0][10:], dict_data[1][1][10:], label=r'+$R_D$', linewidth=3, alpha=1, zorder=10)
plt.plot(minimum_x[1], minimun_y[1], '*', color='#ff7f0e', alpha=1, markersize=22, zorder=15)
plt.plot(dict_data[2][0][10:], dict_data[2][1][10:], label=r'+$R_D$+$R_G$', linewidth=3, alpha=1, zorder=10)
plt.plot(minimum_x[2], minimun_y[2], '*', color='#2ca02c', alpha=1, markersize=22, zorder=15)
plt.plot(dict_data[3][0][10:], dict_data[3][1][10:], label=r'+WD-$R_D$+$R_G$', linewidth=3, alpha=1, zorder=10)
plt.plot(minimum_x[3], minimun_y[3], '*', color='#d62728', alpha=1, markersize=22, zorder=15)
plt.plot(dict_data[4][0][:], dict_data[4][1][:], label=r'+WD-$R_D$+WA-$R_G$', linewidth=3, alpha=1, zorder=10)
plt.plot(minimum_x[4], minimun_y[4], '*', color='#9467bd', alpha=1, markersize=22, zorder=15, label='Minimum Value')
# plt.plot(minimum_x, minimun_y, '*', color='black', alpha=0, markersize=10, label='Minimum Value')
plt.legend(framealpha=1, fontsize=18)
plt.xlabel('Training Steps', fontsize=25)
plt.ylabel('FID50k', fontsize=25) # rotation=0, labelpad=30
plt.title('Pokemon Dataset', fontsize=25)
plt.tight_layout()
plt.show()
print(plt.rcParams['axes.prop_cycle'])

