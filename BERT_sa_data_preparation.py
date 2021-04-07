# %%
import json

with open("C:\\Users\\j-sac\\Dropbox\\Praktisches Seminar\\Daten für AE und ASC\\final_train_ae_CLS.json") as f:
   train_ae = json.load(f)

with open("C:\\Users\\j-sac\\Dropbox\\Praktisches Seminar\\Daten für AE und ASC\\final_train_asc_CLS.json") as f:
   train_asc = json.load(f)

# %%
import pandas as pd
df_train_ae = pd.DataFrame(train_ae).transpose()
sentences_train_ae = df_train_ae.sentence.values
labels_train_ae = df_train_ae.label.values

df_train_asc = pd.DataFrame(train_asc).transpose()

# Problem: Die Daten sind schon tokenized aber BERT nutzt spezielle Tokenization Methoden (padding, trunkate...)
# Gibt es auch eine raw-Version der Daten?
