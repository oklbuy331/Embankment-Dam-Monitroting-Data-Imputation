import numpy as np
import matplotlib.pyplot as plt
from tsl.utils import numpy_metrics

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


def eval_model(y_hat, y_true, mask):
    mae = []
    mse = []
    mre = []
    check_mae = numpy_metrics.masked_mae(y_hat, y_true, mask)
    for i in range(y_true.shape[1]):
        check_mae = np.append(check_mae, numpy_metrics.masked_mae(y_hat[:, i], y_true[:, i], mask[:, i]))
    mae.append(check_mae)

    check_mse = numpy_metrics.masked_mse(y_hat, y_true, mask)
    for i in range(y_true.shape[1]):
        check_mse = np.append(check_mse, numpy_metrics.masked_mse(y_hat[:, i], y_true[:, i], mask[:, i]))
    mse.append(check_mse)

    check_mre = numpy_metrics.masked_mre(y_hat, y_true, mask)
    for i in range(y_true.shape[1]):
        check_mre = np.append(check_mre, numpy_metrics.masked_mre(y_hat[:, i], y_true[:, i], mask[:, i]))
    mre.append(check_mre)
    print(f'Mean MAE: {check_mae[0]:.3f} MSE: {check_mse[0]:.3f} MRE: {check_mre[0]:.4f}')
    print(f'P07 MAE: {check_mae[7]:.3f} MSE: {check_mse[7]:.3f} MRE: {check_mre[7]:.4f}')
    print(f'P08 MAE: {check_mae[8]:.3f} MSE: {check_mse[8]:.3f} MRE: {check_mre[8]:.4f}')
    return mae, mse, mre


# Load model prediction
dir = '/home/oklbuy/PycharmProjects/spin/spin/datasets/'
grin_hat = np.load(dir + 'grin_hat.npy')
bigan_hat = np.load(dir + 'bigan_hat.npy')
brits_hat = np.load(dir + 'brits_hat.npy')
y_true = np.load(dir + 'y_true.npy')
mask = np.load(dir + 'mask.npy')

# Evaluate model
grin_mae, grin_mse, grin_mre = eval_model(grin_hat, y_true, mask)
bigan_mae, bigan_mse, bigan_mre = eval_model(bigan_hat, y_true, mask)
brits_mae, brits_mse, brits_mre = eval_model(brits_hat, y_true, mask)

fig, ax = plt.subplots(1, 1, figsize=[7, 3.5])
sensor1 = 10
sensor2 = 13
x = np.arange(0, len(y_true))

# Ground truth plot
# ax.plot(x, y_true[:, sensor1], linestyle='-.', color='black', alpha=1, lw=1)
# ax.plot(x, y_true[:, sensor2], linestyle='-.', color='black', alpha=1, lw=1)

# Grin plot
ax.plot(x, grin_hat[:, sensor1-1], label='The proposed model', linestyle='-', color='black', alpha=1, lw=1)
ax.plot(x, grin_hat[:, sensor2-1], linestyle='-', color='black', alpha=1, lw=1)

# BiGAN plot
ax.plot(x, bigan_hat[:, sensor1-1], label='BiGAN', linestyle='--', color='black', alpha=0.5, lw=0.8)
ax.plot(x, bigan_hat[:, sensor2-1], linestyle='--', color='black', alpha=0.5, lw=0.8)

# BRITS plot
ax.plot(x, brits_hat[:, sensor1-1], label='BRITS', linestyle=':', color='black', alpha=0.5, lw=0.8)
ax.plot(x, brits_hat[:, sensor2-1], linestyle=':', color='black', alpha=0.5, lw=0.8)

ax.annotate('P'+str(sensor1), xy=(1425, 32.0), xycoords='data', textcoords='offset points', xytext=(30, 30),
            fontsize=9, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
ax.annotate('P'+str(sensor2), xy=(1100, 17.0), xycoords='data', textcoords='offset points', xytext=(20, -40),
            fontsize=9, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2'))

ax.set_xlabel('the number of samples', font=font1, rotation=0), ax.set_ylabel('level/m', font=font1, rotation=90)
ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=4)
plt.savefig('./compare.pdf', dpi=300, bbox_inches='tight')
plt.show()
pass