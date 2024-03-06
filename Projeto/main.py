import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("./dados/Amaro_testeSL.csv", skiprows=[0,1,2,4,5], header=[0,1], nrows=7200)
df = df.rename(columns={'Unnamed: 0_level_0': ''})

selectedColumn = 'Skeleton 018:LFTC'

max = df[selectedColumn, "Z"].max()
min = df[selectedColumn, "Z"].min()
globalMean = (max + min) / 2

pp = [] ## previousPoints
uh = False # Uphill
cs = [[],[]] #cicle start


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
        cs[0].append(index)
        cs[1].append(value)

    if (mean > value and uh and value > globalMean):
        uh = False

fig,ax = plt.subplots()
sns.lineplot(y=df[selectedColumn,"Z"], x = df["","Frame"])
ax.scatter(cs[0], cs[1], c='firebrick', zorder=100)

plt.show()
