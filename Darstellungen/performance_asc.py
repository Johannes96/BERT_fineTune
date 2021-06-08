# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %% data
url = 'https://raw.githubusercontent.com/Johannes96/BERT_fineTune/master/data/finetuning_data_asc.csv'
data_finetune = pd.read_csv(url, sep=';')

acc = data_finetune['accuracy'].tolist()
f1 = data_finetune['f1-score'].tolist()
mse = data_finetune['mse'].tolist()
labels = data_finetune['Modell'].tolist()
# %% accuracy und f1-score asc

# create figure
n_groups = 6
fig, ax1 = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
# opacity = 0.8

rects1 = plt.bar(index, acc, bar_width, color='g', label='Accuracy')

rects2 = plt.bar(index + bar_width, f1, bar_width, color='#8aab33', label='F1-Score')

# plt.xlabel('Sprachmodell')
plt.ylabel('Performance')
plt.title('Performace beim Fine-Tuning der Aspekt Sentiment Klassifikation')
plt.xticks(index + bar_width * 0.5, labels)
plt.legend(loc='upper right')

plt.tight_layout()
# plt.show()
plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/ft_performance_asc_v1.png', dpi=300)

# %% mse asc
n_groups = 6
fig, ax1 = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35

rects1 = plt.bar(index, mse, bar_width, color='#8aab33', label='mse')

# plt.xlabel('Sprachmodell')
plt.ylabel('Mean Squared Error')
plt.title('Performace beim Fine-Tuning der Aspekt Sentiment Klassifikation')
plt.xticks(index, labels) # bei zwei Balken: index + bar_width * 0.5

plt.tight_layout()
plt.show()
# plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/ft_performance_asc_mse_v1.png', dpi=300)
# %% accuracy und f1-score ae

# %% mse ae
# ggf. mse von asc und ae in einer Darstellung kombinieren
