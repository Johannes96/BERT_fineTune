import matplotlib.pyplot as plt
# %%
x = [20000, 40000, 60000, 80000]
y_time = [1509, 3030, 4456, 6078]

y_time_m = []
y_cost = []
for i in y_time:
    y_time_m.append(i /60)
    y_cost.append((i / 3600) * 0.35)

fig, ax1 = plt.subplots()

color = '#8aab33'
ax1.set_xlabel('Anzahl Trainingsdatensätze')
ax1.set_ylabel(ylabel='Trainingskosten [$]')
ax1.plot(x, y_cost, color=color, label='Trainingskosten')
ax1.tick_params(axis='y')
# plt.legend(loc='upper left')
plt.xticks(x, x)

plt.show()
# plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/hyperparametertuning_asc_BERT_v4.png', dpi=300)

# to-do
# erstelle Prediction für höhere Werte