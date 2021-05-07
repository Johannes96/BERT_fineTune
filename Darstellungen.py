# %%
import matplotlib.pyplot as plt
import pandas as pd
import wget
# %%
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