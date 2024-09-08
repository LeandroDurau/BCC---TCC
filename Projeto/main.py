import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import sys

id = sys.argv[1]
axis = sys.argv[2]

lista_pessoas = {
    # '12': 'Katia',
    # '16': 'Nelson',
    # '18': 'Amaro',
    '19': 'Adilson',
    # '22': 'Jair',
    # '25': 'Marilene',
    # '27': 'Izabel',
    # '30': 'Josui',
    # '32': 'Carlos',
    # '34': 'Arnaldo'
}

pessoa = lista_pessoas[id]

df = pd.read_csv("./dados/"+pessoa+"_testeSL.csv", skiprows=[0,1,2,4,5], header=[0,1], nrows=7200)
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

max = df[selectedColumn, axis].max()
min = df[selectedColumn, axis].min()
globalMean = (max + min) / 2

pp = [] ## previousPoints
uh = False # Uphill

cv = [[],[]] #cicle valley
cp = [[],[]] #cicle peak


cPeak = scipy.signal.find_peaks(df[selectedColumn, axis], prominence = 1)
cPeak2 = map(lambda x: cp[0][x], cp[0])

fig,ax = plt.subplots()
sns.lineplot(y=df[selectedColumn,axis], x = df["","Frame"])
ax.scatter(cPeak, 0, c='firebrick', zorder=100)

# ax.scatter(cp[0], cp[1], c='orange', zorder=100)
#
# for i in range(len(cv[0])):
#     ax.annotate(str(i+1), (cp[0][i], cp[1][i] + 1))
#     ax.annotate(str(i+1), (cv[0][i], cv[1][i] - 10))
#
# plt.show()