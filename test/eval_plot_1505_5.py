import matplotlib.pyplot as plt
import numpy as np

config = {
    'figure.figsize': [3.2, 3.2],
    'font.family': 'Times New Roman',
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

fig = plt.figure(6.4, 3.2)
sensor = 10
ax = fig.add_subplot(111)

# Deep learning methods plot
ax.plot(np.arange(0, len(y_true[:, 35, sensor])), y_true[:, 35, sensor], label='y_hat', color='black', alpha=0.6)
ax.scatter(np.where(mask[:, 35, sensor] == 1)[0], y_hat[mask[:, 35, sensor].astype('bool'), 35, sensor], label='y_true', marker='.', color='black')

# Scikit learn methods plot
ax.plot(np.arange(0, len(imputed_mice.iloc[:, sensor])), imputed_mice.iloc[:, sensor], label='y_hat', color='red')
ax.plot(np.arange(0, len(df.iloc[:, sensor])), df.iloc[:, sensor], label='y', color='black')

# MF
ax.plot(np.arange(0, len(imputed_mf[:, sensor])), imputed_mf[:, sensor], label='y_hat', color='red')

ax.legend()



plt.savefig('./eval_plot_1505_5.tiff', dpi=300, bbox_inches='tight')
plt.show()
pass

dist = np.array([[-12, 1280,0],
                 [-3.5, 1280, 0],
                 [8, 1280, 0],
                 [-12, 1290, 0],
                 [-3.5, 1290, 0],
                 [8.5, 1290, 0],
                 [-12, 1305, 0],
                 [-3.5, 1305, 0],
                 [6.5, 1305, 0],
                 [-12, 1280, 55],
                 [-3.5, 1280, 55],
                 [8, 1280, 55],
                 [-12, 1290, 55],
                 [-3.5, 1290, 55],
                 [8.5, 1290, 55],
                 [-12, 1305, 55],
                 [-3.5, 1305, 55],
                 [6.5, 1305, 55],
                 [-2.5, 1265, 0],
                 [2.5, 1265, 0]
                 ])
adj = np.expand_dims(dist, 1).repeat(20, axis=1)
square_sum = np.sum((dist - adj)**2, axis=2)**(1/2)