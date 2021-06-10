import matplotlib.pyplot as plt
import numpy as np
# %%
# load data and create prognoses
x = [20000, 40000, 60000, 80000]
y_time = [1509, 3030, 4456, 6078]

x_prog = [20000 * 5, 20000 * 10, 20000 * 25]
y_time_prog = [1500 * 5, 1500 * 10, 1500 * 25]

# convert time to minutes and calculate costs
y_time_m = []
y_cost = []
y_cost_prog = []
for i in y_time:
    y_time_m.append(i / 60)
    y_cost.append((i / 3600) * 0.35)

for prog in y_time_prog:
    y_cost_prog.append((prog / 3600) * 0.35)

# Create bars
barWidth = 0.8
bars1 = y_cost
bars2 = y_cost_prog
bars3 = bars1 + bars2

# The X position of bars
r1 = [1, 2, 3, 4]
r2 = [5, 6, 7]
r3 = r1 + r2

# Create barplot
plt.bar(r1, bars1, width=barWidth, color='#8aab33')
plt.bar(r2, bars2, width=barWidth, color='g', label='Prognose')

# Create legend and label axes
plt.ylim(0, 4)
plt.legend(loc='upper left')
plt.ylabel('Trainingskosten [$]')
plt.xlabel('Anzahl Trainingsdaten')

# Text below each barplot with a rotation of X°
plt.xticks([r + 1 for r in range(len(r3))],
           ['20.000', '40.000', '60.000', '80.000', '100.000', '0.5 Mio', '1 Mio'], rotation=45)

# Create labels
label = []
for cost in bars3:
    label.append(str(round(cost, 2)) + ' $')

# Text on the top of each bar
for i in range(len(r3)):
    plt.text(x=r3[i] - 0.2, y=bars3[i] + 0.1, s=label[i], size=8, )

# Adjust the margins
plt.subplots_adjust(bottom=0.2, top=0.95)

plt.show()
plt.savefig('/home/johannes/Dropbox/Praktisches Seminar/Darstellungen/ft_datasetsize_variation_v1.png', dpi=300)