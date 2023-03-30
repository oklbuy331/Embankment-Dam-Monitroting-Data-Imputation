import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

config = {
    'figure.figsize': [3.2, 3.2],
    'font.family': 'sans-serif',
    'font.sans-serif': 'Times New Roman',
    'font.size': 9,
    'mathtext.fontset': 'stix',
    'font.weight': '700',
    'axes.labelpad': 1,
    'xaxis.labellocation': 'center',
    'yaxis.labellocation': 'center',
    'xtick.major.pad': 1,
    'ytick.major.pad': 1,
    'legend.frameon': False,
    'legend.labelspacing': 0.1,
    'legend.handletextpad': 0.4,
    'legend.loc': 'upper right',
    'legend.edgecolor': '1',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.unicode_minus': False,
}
plt.rcParams.update(config)

font = {'family': 'Times New Roman',
         'weight': 'regular',
         'style': 'normal',
         'size': 9,
         }

font1 = {'family': 'Times New Roman',
         'weight': 'regular',
         'style': 'normal',
         'size': 9}

# Load prediction and ground truth
dir = '/home/oklbuy/PycharmProjects/spin/spin/datasets/'
grin_hat = np.load(dir + 'grin_hat.npy')
bigan_hat = np.load(dir + 'bigan_hat.npy')
brits_hat = np.load(dir + 'brits_hat.npy')
y_true = np.load(dir + 'y_true.npy')
mask = np.load(dir + 'mask.npy')

fig, ax = plt.subplots(1, 1, figsize=[7, 3.5])
plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)
sensor = 10

# Deep learning methods plot
x = np.arange(0, len(y_true[:, sensor]))
ax.plot(x, y_true[:, sensor], label='Observations', color='black', alpha=0.6)
ax.scatter(np.where(mask[:, sensor] == 1)[0], grin_hat[mask[:, sensor].astype('bool'), sensor],
           label='The proposed model', marker='.', color='black')
ax.scatter(np.where(mask[:, sensor] == 1)[0], bigan_hat[mask[:, sensor].astype('bool'), sensor],
           label='BiGAN', marker='s', s=9, color='black', alpha=0.5)
ax.scatter(np.where(mask[:, sensor] == 1)[0], brits_hat[mask[:, sensor].astype('bool'), sensor],
           label='BRITS', marker='^', s=9, color='black', alpha=0.5)
ax.set_ylim(10.0, 30.6)
ax.set_xlabel('The number of samples', font=font1, rotation=0), ax.set_ylabel('level/m', font=font1, rotation=90)
ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=4)

# 嵌入绘制局部放大图的坐标系
axins1 = inset_axes(ax, width="25%", height="40%", loc='lower left', bbox_to_anchor=(0.3, 0.03, 1, 1),
                   bbox_transform=ax.transAxes)
axins1.xaxis.set_major_locator(plt.NullLocator())

# 在子坐标系中绘制原始数据
axins1.plot(x, y_true[:, sensor], label='Observations', color='black', alpha=0.6)
axins1.scatter(np.where(mask[:, sensor] == 1)[0], grin_hat[mask[:, sensor].astype('bool'), sensor],
           label='The proposed model', marker='.', color='black')
axins1.scatter(np.where(mask[:, sensor] == 1)[0], bigan_hat[mask[:, sensor].astype('bool'), sensor],
           label='BiGAN', marker='s', s=9, color='black', alpha=0.5)
axins1.scatter(np.where(mask[:, sensor] == 1)[0], brits_hat[mask[:, sensor].astype('bool'), sensor],
           label='BRITS', marker='^', s=9, color='black', alpha=0.5)

# X轴的显示范围
xlim0 = 830
xlim1 = 890

# Y轴的显示范围
ylim0 = 25
ylim1 = 29

# 调整子坐标系的显示范围
axins1.set_xlim(xlim0, xlim1)
axins1.set_ylim(ylim0, ylim1)

# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
mark_inset(ax, axins1, loc1=2, loc2=1, fc="none", ec='k', lw=0.6)

# 嵌入绘制局部放大图的坐标系
axins2 = inset_axes(ax, width="25%", height="40%", loc='lower left', bbox_to_anchor=(0.7, 0.03, 1, 1),
                   bbox_transform=ax.transAxes)
axins2.xaxis.set_major_locator(plt.NullLocator())

# 在子坐标系中绘制原始数据
axins2.plot(x, y_true[:, sensor], label='Observations', color='black', alpha=0.6)
axins2.scatter(np.where(mask[:, sensor] == 1)[0], grin_hat[mask[:, sensor].astype('bool'), sensor],
           label='The proposed model', marker='.', s=9, color='black')
axins2.scatter(np.where(mask[:, sensor] == 1)[0], bigan_hat[mask[:, sensor].astype('bool'), sensor],
           label='BiGAN', marker='s', s=9, color='black', alpha=0.5)
axins2.scatter(np.where(mask[:, sensor] == 1)[0], brits_hat[mask[:, sensor].astype('bool'), sensor],
           label='BRITS', marker='^', s=9, color='black', alpha=0.5)

# X轴的显示范围
xlim0 = 2040
xlim1 = 2120

# Y轴的显示范围
ylim0 = 22.5
ylim1 = 26

# 调整子坐标系的显示范围
axins2.set_xlim(xlim0, xlim1)
axins2.set_ylim(ylim0, ylim1)

# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
mark_inset(ax, axins2, loc1=2, loc2=1, fc="none", ec='k', lw=0.6)

plt.savefig('./inset.pdf', dpi=300, bbox_inches='tight')

# 显示
plt.show()

pass