import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import missingno as msno

path = '/home/oklbuy/PycharmProjects/spin/spin/datasets/data.csv'
df = pd.read_csv(path, index_col=0)
df.index = pd.to_datetime(df.index)

config = {
    'figure.figsize': [3.2, 3.2],
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
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

ax0 = msno.matrix(df, figsize=(3.6, 3.2), fontsize=9)
ax0.yaxis.set_major_locator(plt.NullLocator())
plt.savefig('./missingplot_fig.pdf', dpi=300, bbox_inches='tight')
plt.show()
pass

dist = np.array([[-12, 1280,0],
                 [-3.5, 1280, 0],
                 [8, 1280, 0],
                 [-12, 1290, 0],
                 [-3.5, 1290, 0],
                 [8, 1290, 0],
                 [-12, 1305, 0],
                 [-3.5, 1305, 0],
                 [6.5, 1305, 0],
                 [-12, 1280, 55],
                 [-3.5, 1280, 55],
                 [8, 1280, 55],
                 [-12, 1290, 55],
                 [-3.5, 1290, 55],
                 [8, 1290, 55],
                 [-12, 1305, 55],
                 [-3.5, 1305, 55],
                 [6.5, 1305, 55],
                 [-2.5, 1265, 0],
                 [2.5, 1265, 0]
                 ])
adj = np.expand_dims(dist, 1).repeat(20, axis=1)
square_sum = np.sum((dist - adj)**2, axis=2)**(1/2)