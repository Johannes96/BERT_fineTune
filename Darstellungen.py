# %%
import matplotlib.pyplot as plt
import pandas as pd
import wget
import numpy as np
# %% scatterplots pretraining
url = 'https://raw.githubusercontent.com/Johannes96/BERT_fineTune/master/data/Pretraining_data.csv'

data_pretrain = pd.read_csv(url, sep=';')
# %% create Dataset for GLUE by dropping not used scores and remove rows with empty values
df_GLUE = data_pretrain.drop(['SuperGLUE', 'SQuAD 2.0', 'RACE'], axis=1)
df_GLUE = df_GLUE.dropna()
df_GLUE['Kosten'] = df_GLUE['Kosten']
# Entferne Roberta weil die Kosten sehr hoch sind und es die Darstellung verzerrt
df_GLUE.drop(df_GLUE.loc[df_GLUE['Modell']=='RoBERTa'].index, inplace=True)
# %% Erstelle arrays zum plotten
x = data_pretrain['Kosten']
y = data_pretrain['GLUE']
param = data_pretrain['Parameter']
param_norm = list(e * 0.00001 for e in param) # verändere Größe der Punkte
labels = data_pretrain['Modell']

x_max = df_GLUE['Kosten'].max()
x_min = df_GLUE['Kosten'].min()

y_max = df_GLUE['GLUE'].max()
y_min = df_GLUE['GLUE'].min()
print(x_max, x_min)
# %%
plt.scatter(x, y, label='GLUE', s=param_norm, alpha=0.7, c='#8aab33')

plt.xlabel('Trainingskosten')
plt.ylabel('Performance')
plt.title('Pretraining Effizienz - GLUE')
plt.xlim(x_max + 2000, 0) # erst max weil weniger Kosten besser sind
plt.ylim(y_min - 0.2, y_max + 0.2)

for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]))

plt.show()
# %% Speichere Darstellung in Dropbox
plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/pretraining_effizienz_GLUE.png', dpi=300)

# %% linegraph hyperparametertuning - Epochs
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # epochs
y_eff = [1, 0.5191074356, 0.3223776789, 0.2088268125, 0.1447251207, 0.09251209684, 0.05662804618, 0.03243850918, 0.01853784435, 0] # efficiency (accuracy/trainingtime)
y_acc = [0.443, 0.529, 0.560, 0.567, 0.576, 0.571, 0.563, 0.563, 0.566, 0.559]
y_mse = [1, 0.3063763608, 0.09797822706, 0, 0.01244167963, 0.1461897356, 0.2052877138, 0.03110419907, 0.231726283, 0.1726283048]
plt.plot(x, y_acc, color='#8aab33',label='Accuracy')
plt.plot(x, y_eff,
         color='#8aab33',
         # linewidth=1.0,
         linestyle='--',
         label='Efficiency'
        )
plt.plot(x, y_mse,
         color='#8aab33',
         # linewidth=1.0,
         linestyle=':',
         label='MSE'
        )
plt.xlabel('Epochs')
# plt.ylabel('Performance bzw. Effizienz')
plt.title('Hyperparametertuning - aspectbased sentiment analysis BERT')
plt.legend(loc='upper right')
plt.show()

# %%
plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/hyperparametertuning_asc_BERT.png', dpi=300)

# %%
# Create some mock data
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
ax1.plot(x, y_mse,
         color='#8aab33',
         linestyle='--',
         label='MSE'
        )
plt.plot(x, y_eff,
         color='#8aab33',
         linestyle=':',
         label='Efficiency'
        )
ax1.tick_params(axis='y', labelcolor=color)
plt.legend(loc='upper left')
plt.xticks(x, x)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Trainingszeit [min]', color=color)  # we already handled the x-label with ax1
ax2.plot(x, y_time_m, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/hyperparametertuning_asc_BERT_v4.png', dpi=300)

# %% scatter-plot finetuning-asc
url = 'https://raw.githubusercontent.com/Johannes96/BERT_fineTune/master/data/finetuning_data.csv'
data_finetune = pd.read_csv(url, sep=';')

labels2 = data_finetune['Modell']
acc_asc = data_finetune['accuracy']
time_asc = data_finetune['trainingtime in s'] / 60
params = data_finetune['parameters'] * 0.000001
x_max = time_asc.max() + 1
x_min = time_asc.min()

plt.scatter(time_asc, acc_asc, label='GLUE', s=params, alpha=0.7, c='#8aab33')

plt.xlabel('Trainingszeit [min]')
plt.ylabel('Performance [accuracy]')
# plt.title('finetuning Effizienz - ASC')
plt.xlim(x_max, 4) # erst max weil weniger Zeit besser ist
# plt.ylim(y_min - 0.2, y_max + 0.2)

for i, label in enumerate(labels2):
    plt.annotate(label, (time_asc[i], acc_asc[i]))

# plt.show()
plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/finetuning_effizienz_asc.png', dpi=300)
