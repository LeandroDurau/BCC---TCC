import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#df = pd.read_csv("./dados/Amaro_testeSL.csv", skiprows=[0,1,2,4,5], header=[0,1], nrows=7200)
df = pd.read_csv("./dados/Marilene_testeSL.csv", skiprows=[0,1,2,4,5], header=[0,1], nrows=7200)
df = df.rename(columns={'Unnamed: 0_level_0': ''})

columns = [
        'Skeleton 025:LFTC', #0
        'Skeleton 025:RIPS', #1
        'Skeleton 025:Head', #2
        'Skeleton 025:LShoulder', #3
        'Skeleton 025:LUArm', #4
        'Skeleton 025:LFLE', #5
        ]

selectedColumn = columns[0]
selectedColumn2 = columns[5]

########################################################

max = df[selectedColumn, "Z"].max()
min = df[selectedColumn, "Z"].min()
globalMean = (max + min) / 2

pp = [] ## previousPoints
uh = False # Uphill
cv = [[],[]] #cicle valley
cp = [[],[]] #cicle peak


for index in range(len(df[selectedColumn,"Z"])):
    value = df[selectedColumn,"Z"][index]
    pp.append(value)

    if len(pp) > 30 :
        pp.pop(0)
    elif len(pp) < 30:
        continue
    
    mean = np.mean(pp)
    if (mean < value and not uh and value < globalMean):
        uh = True
        cv[0].append(index)
        cv[1].append(df[selectedColumn2,"Z"][index])

    if (mean > value and uh and value > globalMean):
        uh = False
        cp[0].append(index)
        cp[1].append(df[selectedColumn2,"Z"][index])


fig,ax = plt.subplots()
sns.lineplot(y=df[selectedColumn2,"Z"], x = df["","Frame"])
ax.scatter(cv[0], cv[1], c='firebrick', zorder=100)
ax.scatter(cp[0], cp[1], c='orange', zorder=100)

plt.show()
