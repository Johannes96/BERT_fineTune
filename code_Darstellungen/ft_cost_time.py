import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %% cost and time asc

url = 'https://raw.githubusercontent.com/Johannes96/BERT_fineTune/master/data/finetuning_data_asc.csv'
df = pd.read_csv(url, sep=';')
df.rename(columns={'trainingtime in s':'trainingtime'}, inplace=True)
df['trainingtime'] = df['trainingtime'] / 60

fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

labels = df['Modell']
x = np.arange(len(labels))
width = 0.3

df.trainingtime.plot(kind='bar', color='#8aab33', ax=ax, width=width, position=1, label='Trainingszeit')
df.cost.plot(kind='bar', color='green', ax=ax2, width=width, position=0, label='Trainingskosten')

ax.set_ylabel('Trainingszeit [min]')
ax2.set_ylabel('Trainingskosten [$]')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.set_xlim(-0.5)  # create space between left y-axis and first bar
fig.tight_layout()
fig.legend(loc='upper left')

plt.show()
# plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/ft_time_cost_asc.png', dpi=300)

# %% cost and time ae

url = 'https://raw.githubusercontent.com/Johannes96/BERT_fineTune/master/data/finetuning_data_ae.csv'
df = pd.read_csv(url, sep=';')
df.rename(columns={'trainingtime in s':'trainingtime'}, inplace=True)  # rename column so that it has no whitespaces
df['trainingtime'] = df['trainingtime'] / 60  # convert sec to min

# configure labels so that the size of model is in subscript (tiefgestellt)
labels = df['Modell'].tolist()
labels_v2 = []
i = 0
for model in labels:
    model_v2 = model.split('-', 1)
    # print(model_v2[0])
    labels_v2.append(f'${model_v2[0]}_')
    labels_v2[i] += '{'
    labels_v2[i] += model_v2[1]
    labels_v2[i] += '}$'
    i += 1
df.replace(labels, labels_v2, inplace=True)

fig = plt.figure()  # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

labels = df['Modell']
x = np.arange(len(labels))
width = 0.3

df.trainingtime.plot(kind='bar', color='#8aab33', ax=ax, width=width, position=1, label='Trainingszeit')
df.Kosten.plot(kind='bar', color='green', ax=ax2, width=width, position=0, label='Trainingskosten')

ax.set_ylabel('Trainingszeit [min]')
ax2.set_ylabel('Trainingskosten [$]')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.set_xlim(-0.5)
fig.tight_layout()
fig.legend(loc='upper left')

plt.show()
# plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/ft_time_cost_ae.png', dpi=300)
