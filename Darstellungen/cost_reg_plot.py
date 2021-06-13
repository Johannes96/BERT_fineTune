import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

url = 'https://raw.githubusercontent.com/Johannes96/BERT_fineTune/master/data/cost_reg_plot.csv'
data = pd.read_csv(url, sep=',')

data_asc = data.drop(['ft_cost_ae'], axis=1)
data_asc.dropna(inplace=True)

data_ae = data.drop(['ft_cost_asc'], axis=1)
data_ae.dropna(inplace=True)

sns.regplot(x='ft_cost_asc', y='pt_cost', data=data_asc)
plt.show()

# Ergebnis --> kein Zusammenhang