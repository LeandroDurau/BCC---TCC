import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

id = sys.argv[1]
axis = sys.argv[2]


lista_pessoas = {
    '12': 'Katia',
    '16': 'Nelson',
    '18': 'Amaro',
    '19': 'Adilson',
    '22': 'Jair',
    '25': 'Marilene',
    '27': 'Izabel',
    '30': 'Josui',
    '32': 'Carlos',
    '34': 'Arnaldo'
}

pessoa = lista_pessoas[id]

df = pd.read_csv("./dados/"+pessoa+"_testeSL.csv", skiprows=[0,1,2,4,5], header=[0,1], nrows=14400)
df = df.rename(columns={'Unnamed: 0_level_0': ''})

columns = [
        'Skeleton 0'+id+':LFTC', #0
        'Skeleton 0'+id+':RIPS', #1
        'Skeleton 0'+id+':Head', #2
        'Skeleton 0'+id+':LShoulder', #3
        'Skeleton 0'+id+':LUArm', #4
        'Skeleton 0'+id+':LFLE', #5
        ]

selectedColumn = columns[0]
selectedColumn2 = columns[0]

########################################################
# print(df.head())
# print('\n\n')
# print(df.keys())
# print('\n\n')
max = df[selectedColumn, axis].max()
min = df[selectedColumn, axis].min()
globalMean = (max + min) / 2

pp = [] ## previousPoints
uh = False # Uphill

cv = [[],[]] #cicle valley
cp = [[],[]] #cicle peak


for index in range(len(df[selectedColumn,axis])):
    value = df[selectedColumn,axis][index]
    pp.append(value)

    if len(pp) > 25 :
        pp.pop(0)
    elif len(pp) < 25:
        continue
   
    mean = np.mean(pp)
    if (mean < value and not uh and value < globalMean):
        uh = True
        cv[0].append(index)
        cv[1].append(df[selectedColumn2,axis][index])

    if (mean > value and uh and value > globalMean):
        uh = False
        cp[0].append(index)
        cp[1].append(df[selectedColumn2,axis][index])


fig,ax = plt.subplots()
sns.lineplot(y=df[selectedColumn2,axis], x = df["","Frame"])
ax.scatter(cv[0], cv[1], c='firebrick', zorder=100)
ax.scatter(cp[0], cp[1], c='orange', zorder=100)

for i in range(len(cv[0])):
    ax.annotate(str(i+1), (cp[0][i], cp[1][i] + 1))
    ax.annotate(str(i+1), (cv[0][i], cv[1][i] - 10))

plt.show()
