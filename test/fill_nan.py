import matplotlib.pyplot as plt
import numpy as np

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

fig = plt.figure()
ax = fig.add_subplot(111)
sensor = 9
ax.plot(np.arange(0, len(y_h[:, sensor])), y_h[:, sensor], label='y_hat', color='black', alpha=0.5)
ax.plot(np.arange(0, len(y_t[:, sensor])), y_t[:, sensor], label='y_true', color='black')
ax.legend()

plt.savefig('./fill_nan.tiff', dpi=300, bbox_inches='tight')
plt.show()
pass