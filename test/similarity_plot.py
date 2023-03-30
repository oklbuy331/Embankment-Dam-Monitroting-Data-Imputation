import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from sklearn.feature_selection import mutual_info_regression

config = {
    'figure.figsize': [3.2, 3.2],
    'font.family': 'serif',
    'font.serif': 'SimSun',
    'font.sans-serif': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'font.weight': '700',
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.labelpad': 1,
    'xtick.major.pad': 1,
    'ytick.major.pad': 1,
    'legend.frameon': False,
    'legend.labelspacing': 0.1,
    'legend.handletextpad': 0.4,
    'legend.loc': 'upper right',
    'legend.edgecolor': '1',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'font.size': 9,
    'axes.unicode_minus': False,
}
plt.rcParams.update(config)
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'style': 'normal',
         'size': 9,
         }

y_h = y_hat[:, 35, :]
y_t = y_true[:, 35, :]

mi = []
for i in range(20):
    mi.append(mutual_info_regression(y_h, y_h[:, i]))
mi = np.concatenate(mi).reshape(20, -1)
for i in range(len(mi)):
    min = mi[i].min();
    max = mi[i].max()
    mi[i] = (mi[i] - min) / (max - min)

plt.style.use('ggplot')
fig= plt.figure(constrained_layout=True, figsize=(6.4, 6.4))
nrows, ncols = 2, 2
gspec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig, width_ratios=[1, 1])

ax = plt.subplot(gspec[0, 0])
im = ax.imshow(dist, vmin=0, vmax=1, cmap='gray_r', aspect=1)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.spines.clear()

ax = plt.subplot(gspec[0, 1])
im = ax.imshow(linear_corr, vmin=0, vmax=1, cmap='gray_r', aspect=1)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.spines.clear()

plt.savefig('./similarity_plot.tiff', dpi=300, bbox_inches='tight')
plt.show()
pass