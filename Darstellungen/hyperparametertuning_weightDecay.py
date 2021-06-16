# %%
import matplotlib.pyplot as plt
import pandas as pd
# %% linegraph hyperparametertuning - Epochs (mit zwei y-Achsen)
x = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
y_acc = [0.553, 0.549, 0.554, 0.549, 0.548, 0.544, 0.553, 0.549, 0.551]
y_mse = [1.215, 1.195, 1.228, 1.232, 1.265, 1.264, 1.240, 1.238, 1.225]
y_time = [529, 532, 533, 533, 532, 534, 533, 534, 536]

y_time_m = []
for i in y_time:
    i_m = i / 60
    y_time_m.append(i_m)

fig, ax1 = plt.subplots()

color = '#8aab33'
ax1.set_xlabel('weight-decay')
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
# plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/hyperparametertuning_weightDecay.png', dpi=300)
