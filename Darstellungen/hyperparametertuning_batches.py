# %%
import matplotlib.pyplot as plt
import pandas as pd
# %% linegraph hyperparametertuning - Epochs (mit zwei y-Achsen)
x = ['16', '32', '64', '128']
y_acc = [0.549, 0.530, 0.503, 0.463]
y_mse = [1.195, 1.410, 1.467, 1.713]
y_time = [532, 451, 407, 387]

y_time_m = []
for i in y_time:
    i_m = i / 60
    y_time_m.append(i_m)

fig, ax1 = plt.subplots()

color = '#8aab33'
ax1.set_xlabel('batch-size')
ax1.set_ylabel(ylabel='Performance', color=color)
ax1.plot(x, y_acc, color=color, label='Accuracy')
ax1.plot(x, y_mse, color=color, linestyle='--', label='MSE')
# plt.plot(x, y_eff, color=color, linestyle=':', label='Efficiency')
ax1.tick_params(axis='y', labelcolor=color)
plt.legend(loc='upper left')
plt.xticks(x, x)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Trainingszeit [min]', color=color)  # we already handled the x-label with ax1
ax2.plot(x, y_time_m, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()
# plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/hyperparametertuning_batches.png', dpi=300)
