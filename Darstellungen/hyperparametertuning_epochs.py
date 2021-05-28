# %%
import matplotlib.pyplot as plt
import pandas as pd
# %% linegraph hyperparametertuning - Epochs (alte Version)
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # epochs
y_eff = [1, 0.5191074356, 0.3223776789, 0.2088268125, 0.1447251207, 0.09251209684, 0.05662804618, 0.03243850918, 0.01853784435, 0] # efficiency (accuracy/trainingtime)
y_acc = [0.443, 0.529, 0.560, 0.567, 0.576, 0.571, 0.563, 0.563, 0.566, 0.559]
y_mse = [1, 0.3063763608, 0.09797822706, 0, 0.01244167963, 0.1461897356, 0.2052877138, 0.03110419907, 0.231726283, 0.1726283048]

plt.plot(x, y_acc, color='#8aab33',label='Accuracy')
plt.plot(x, y_eff, color='#8aab33', linestyle='--', label='Efficiency')
plt.plot(x, y_mse, color='#8aab33', linestyle=':', label='MSE')
plt.xlabel('Epochs')
# plt.ylabel('Performance bzw. Effizienz')
plt.title('Hyperparametertuning - aspectbased sentiment analysis BERT')
plt.legend(loc='upper right')

plt.show()
# plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/hyperparametertuning_asc_BERT.png', dpi=300)

# %% linegraph hyperparametertuning - Epochs (neue Version, mit zwei y-Achsen)
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_eff = [1, 0.5191074356, 0.3223776789, 0.2088268125, 0.1447251207, 0.09251209684, 0.05662804618, 0.03243850918, 0.01853784435, 0] # efficiency (accuracy/trainingtime)
y_acc = [0.443, 0.529, 0.560, 0.567, 0.576, 0.571, 0.563, 0.563, 0.566, 0.559]
y_mse = [1, 0.3063763608, 0.09797822706, 0, 0.01244167963, 0.1461897356, 0.2052877138, 0.03110419907, 0.231726283, 0.1726283048]
y_time = [176, 363, 547, 733, 911, 1104, 1285, 1463, 1598, 1784]

y_time_m = []
for i in y_time:
    i_m = i / 60
    y_time_m.append(i_m)

fig, ax1 = plt.subplots()

color = '#8aab33'
ax1.set_xlabel('Epochs')
ax1.set_ylabel(ylabel='', color=color)
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
# plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/hyperparametertuning_asc_BERT_v4.png', dpi=300)
