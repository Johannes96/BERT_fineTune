# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %% scatter asc
url = 'https://raw.githubusercontent.com/Johannes96/BERT_fineTune/master/data/finetuning_data.csv'
data_finetune = pd.read_csv(url, sep=';')

acc = data_finetune['accuracy'].tolist()
f1 = data_finetune['f1-score'].tolist()
mse = data_finetune['mse'].tolist()
labels = data_finetune['Modell'].tolist()

# %%
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

# %%

# data to plot
n_groups = 6

# create plot
fig, ax1 = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
# opacity = 0.8

rects1 = plt.bar(index, acc, bar_width,
color='g',
label='Accuracy')

rects2 = plt.bar(index + bar_width, f1, bar_width,
color='#8aab33',
label='F1-Score')

# plt.xlabel('Sprachmodell')
plt.ylabel('Performance')
plt.title('Performace beim Fine-Tuning der Aspekt Sentiment Klassifikation')
plt.xticks(index + bar_width, labels)
plt.legend()

ax2 = ax1.twinx()

plt.tight_layout()
plt.show()

# %%
print(type(labels))