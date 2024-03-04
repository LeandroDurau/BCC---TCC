import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("./dados/Amaro_testeSL.csv", skiprows=[0,1,2,4,5], header=[0,1], nrows=7200)
df = df.rename(columns={'Unnamed: 0_level_0': ''})

pp = [] # previousPoints
uh = False # Uphill
mp = np.mean(df["Skeleton 018:RIPS","Z"]) #midpoint
cs = [[],[]] #cycle start (for valleys)
ps = [[],[]] #peak start (for hills)

for index in range(len(df["Skeleton 018:RIPS","Z"])):
    value = df["Skeleton 018:RIPS","Z"][index]
    pp.append(value)

    if len(pp) > 15:
        pp.pop(0)
    elif len(pp) < 15:
        continue
    
    mean = np.mean(pp)
    if mean < value and not uh and value < mp: 
        uh = True
        cs[0].append(index)
        cs[1].append(value)
    elif mean > value and uh and value > mp:
        uh = False
        ps[0].append(index)
        ps[1].append(value)

fig, ax = plt.subplots()
sns.lineplot(y=df["Skeleton 018:RIPS","Z"], x=df["","Frame"])
ax.scatter(cs[0], cs[1], c='firebrick', zorder=100)
ax.scatter(ps[0], ps[1], c='royalblue', zorder=100)

plt.show()
