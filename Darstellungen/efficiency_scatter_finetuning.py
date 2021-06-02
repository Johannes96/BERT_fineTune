# %%
import matplotlib.pyplot as plt
import pandas as pd
# %% scatter asc
url = 'https://raw.githubusercontent.com/Johannes96/BERT_fineTune/master/data/finetuning_data.csv'
data_finetune = pd.read_csv(url, sep=';')

labels2 = data_finetune['Modell']
acc_asc = data_finetune['accuracy']
time_asc = data_finetune['trainingtime in s'] / 60
params1 = data_finetune['parameters'] * 0.000001
x_max = time_asc.max() + 1
x_min = time_asc.min()

plt.scatter(time_asc, acc_asc, label='GLUE', s=params1, alpha=0.7, c='#8aab33')

plt.xlabel('Trainingszeit [min]')
plt.ylabel('Performance [accuracy]')
# plt.title('finetuning Effizienz - ASC')
plt.xlim(x_max, 4) # erst max weil weniger Zeit besser ist
# plt.ylim(y_min - 0.2, y_max + 0.2)

for i, label in enumerate(labels2):
    plt.annotate(label, (time_asc[i], acc_asc[i]))

plt.show()
# plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/finetuning_effizienz_asc.png', dpi=300)

# %% scatter ae
