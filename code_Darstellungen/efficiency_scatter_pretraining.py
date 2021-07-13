# %%
import matplotlib.pyplot as plt
import pandas as pd
# %% load pretraining-data
url = 'https://raw.githubusercontent.com/Johannes96/BERT_fineTune/master/data/Pretraining_data.csv'

data_pretrain = pd.read_csv(url, sep=';')
data_pretrain.rename(columns={"SQuAD 2.0": "SQuAD"}, inplace=True)
# %% GLUE

df_GLUE = data_pretrain.drop(['SuperGLUE', 'SQuAD', 'RACE'], axis=1)
df_GLUE = df_GLUE.dropna()
df_GLUE['Kosten'] = df_GLUE['Kosten']

# Entferne nicht benötigte Modelle (weil kein fine-tuning dafür durchgeführt)
df_GLUE.drop(df_GLUE.loc[df_GLUE['Modell']=='RoBERTa'].index, inplace=True)
df_GLUE.drop(df_GLUE.loc[df_GLUE['Modell']=='Bort'].index, inplace=True)
df_GLUE.drop(df_GLUE.loc[df_GLUE['Modell']=='ELECTRA-small'].index, inplace=True)
df_GLUE.reset_index(drop=True, inplace=True)

x = df_GLUE['Kosten']
y = df_GLUE['GLUE']
param = df_GLUE['Parameter']
param_norm = list(e * 0.00001 for e in param) # verändere Größe der Punkte
labels = df_GLUE['Modell']

x_max = df_GLUE['Kosten'].max()
x_min = df_GLUE['Kosten'].min()

y_max = df_GLUE['GLUE'].max()
y_min = df_GLUE['GLUE'].min()
print(x_max, x_min)

plt.scatter(x, y, label='GLUE', s=param_norm, alpha=0.7, c='#8aab33')

plt.xlabel('Trainingskosten')
plt.ylabel('Performance')
plt.title('Pretraining Effizienz - GLUE')
plt.xlim(x_max + 2000, 0) # erst max weil weniger Kosten besser sind
plt.ylim(y_min - 0.2, y_max + 0.2)

for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]))

plt.show()
# plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/pretraining_effizienz_GLUE_v2.png', dpi=300)

# %% SuperGLUE
df_SuperGLUE = data_pretrain.drop(['GLUE', 'SQuAD', 'RACE'], axis=1)
df_SuperGLUE = df_SuperGLUE.dropna()
df_SuperGLUE['Kosten'] = df_SuperGLUE['Kosten']

# Entferne nicht benötigte Modelle
df_SuperGLUE.drop(df_SuperGLUE.loc[df_SuperGLUE['Modell']=='Bort'].index, inplace=True)
df_SuperGLUE.drop(df_SuperGLUE.loc[df_SuperGLUE['Modell']=='DeBERTa-1.5Mrd'].index, inplace=True)
df_SuperGLUE.drop(df_SuperGLUE.loc[df_SuperGLUE['Modell']=='ELECTRA-small'].index, inplace=True)
df_SuperGLUE.reset_index(drop=True, inplace=True)

x = df_SuperGLUE['Kosten']
y = df_SuperGLUE['SuperGLUE']
param = df_SuperGLUE['Parameter']
param_norm = list(e * 0.00001 for e in param) # verändere Größe der Punkte
labels = df_SuperGLUE['Modell']

x_max = df_SuperGLUE['Kosten'].max()
x_min = df_SuperGLUE['Kosten'].min()

y_max = df_SuperGLUE['SuperGLUE'].max()
y_min = df_SuperGLUE['SuperGLUE'].min()
print(x_max, x_min)

plt.scatter(x, y, label='SuperGLUE', s=param_norm, alpha=0.7, c='#8aab33')

plt.xlabel('Trainingskosten')
plt.ylabel('Performance')
plt.title('Pretraining Effizienz - SuperGLUE')
plt.xlim(x_max + 2000, 0) # erst max weil weniger Kosten besser sind
plt.ylim(y_min - 0.1, y_max + 0.2)

for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]))

plt.show()
# plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/pretraining_effizienz_SuperGLUE_v1.png', dpi=300)

# %% SQuAD
df_SQuAD = data_pretrain.drop(['GLUE', 'SuperGLUE', 'RACE'], axis=1)
df_SQuAD = df_SQuAD.dropna()
df_SQuAD['Kosten'] = df_SQuAD['Kosten']

# Entferne nicht benötigte Modelle
df_SQuAD.drop(df_SQuAD.loc[df_SQuAD['Modell']=='DeBERTa-large'].index, inplace=True)
df_SQuAD.drop(df_SQuAD.loc[df_SQuAD['Modell']=='ALBERT-xxlarge'].index, inplace=True)
df_SQuAD.drop(df_SQuAD.loc[df_SQuAD['Modell']=='XLNet'].index, inplace=True)
df_SQuAD.reset_index(drop=True, inplace=True)

x = df_SQuAD['Kosten']
y = df_SQuAD['SQuAD']
param = df_SQuAD['Parameter']
param_norm = list(e * 0.00001 for e in param) # verändere Größe der Punkte
labels = df_SQuAD['Modell']

x_max = df_SQuAD['Kosten'].max()
x_min = df_SQuAD['Kosten'].min()

y_max = df_SQuAD['SQuAD'].max()
y_min = df_SQuAD['SQuAD'].min()
print(x_max, x_min)

plt.scatter(x, y, label='SQuAD', s=param_norm, alpha=0.7, c='#8aab33')

plt.xlabel('Trainingskosten')
plt.ylabel('Performance')
plt.title('Pretraining Effizienz - SQuAD')
plt.xlim(x_max + 2000, 0) # erst max weil weniger Kosten besser sind
plt.ylim(y_min - 0.2, y_max + 0.2)

for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]))

plt.show()
# plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/pretraining_effizienz_SQuAD_v1.png', dpi=300)

# %% RACE
df_RACE = data_pretrain.drop(['GLUE', 'SuperGLUE', 'SQuAD'], axis=1)
df_RACE = df_RACE.dropna()
df_RACE['Kosten'] = df_RACE['Kosten']

# Entferne nicht benötigte Modelle
df_RACE.drop(df_RACE.loc[df_RACE['Modell']=='DeBERTa-large'].index, inplace=True)
df_RACE.drop(df_RACE.loc[df_RACE['Modell']=='Bort'].index, inplace=True)
df_RACE.drop(df_RACE.loc[df_RACE['Modell']=='ALBERT-xxlarge'].index, inplace=True)
df_RACE.drop(df_RACE.loc[df_RACE['Modell']=='XLNet'].index, inplace=True)
df_RACE.reset_index(drop=True, inplace=True)

x = df_RACE['Kosten']
y = df_RACE['RACE']
param = df_RACE['Parameter']
param_norm = list(e * 0.00001 for e in param) # verändere Größe der Punkte
labels = df_RACE['Modell']

x_max = df_RACE['Kosten'].max()
x_min = df_RACE['Kosten'].min()

y_max = df_RACE['RACE'].max()
y_min = df_RACE['RACE'].min()
print(x_max, x_min)

plt.scatter(x, y, label='RACE', s=param_norm, alpha=0.7, c='#8aab33')

plt.xlabel('Trainingskosten')
plt.ylabel('Performance')
plt.title('Pretraining Effizienz - RACE')
plt.xlim(x_max + 2000, 0) # erst max weil weniger Kosten besser sind
plt.ylim(y_min - 0.05, y_max + 0.05)

for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]))

plt.show()
# plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/pretraining_effizienz_RACE_v1.png', dpi=300)
