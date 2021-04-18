# %%
import json

# Opening JSON file
with open('./data/final_train_asc_CLS.json') as json_file:
    train_asc = json.load(json_file)

with open('./data/final_dev_asc_CLS.json') as json_file:
    dev_asc = json.load(json_file)

# %%
train_texts = []
train_labels_temp = []
for i in train_asc:
    train_labels_temp.append(train_asc[i]['polarity'])
    train_texts.append(train_asc[i]['sentence'])

dev_texts = []
dev_labels_temp = []
for i in dev_asc:
    dev_labels_temp.append(dev_asc[i]['polarity'])
    dev_texts.append(dev_asc[i]['sentence'])
# %% Ersetzte labels durch Nummern von 0 bis 4
train_labels = []
for i in train_labels_temp:
    if i == 'str_negative':
        train_labels.append(0)
    elif i == 'negative':
        train_labels.append(1)
    elif i == 'neutral':
        train_labels.append(2)
    elif i == 'positive':
        train_labels.append(3)
    elif i == 'str_positive':
        train_labels.append(4)

dev_labels = []
for i in dev_labels_temp:
    if i == 'str_negative':
        dev_labels.append(0)
    elif i == 'negative':
        dev_labels.append(1)
    elif i == 'neutral':
        dev_labels.append(2)
    elif i == 'positive':
        dev_labels.append(3)
    elif i == 'str_positive':
        dev_labels.append(4)

# %% Erstelle Validationsdaten
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

# %% Load Tokenizer
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# %%
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(dev_texts, truncation=True, padding=True) # changed name to dev_texts

# %% erstelle Datensätze
import torch

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, dev_labels)
# %% adjust BERT so that he can work with 5 labels (y)
from transformers import BertForSequenceClassification, BertConfig, Trainer, TrainingArguments
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 5 # adjust to change number of distinct y (labels)
model = BertForSequenceClassification(config)
# print(model.parameters) # show how many labels (out_features = ___) are defined

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)
# %%
trainer.train()