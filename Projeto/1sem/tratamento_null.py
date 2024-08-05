import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#katia - 12
#nelson 16
#Amaro - 18
#Adilson - 19
#Jair - 22
#Marilene - 25
#Izabel 27
#Josui - 30
#Carlos - 32
#Arnaldo - 34

#df = pd.read_csv("./dados/Amaro_testeSL.csv", skiprows=[0,1,2,4,5], header=[0,1], nrows=7200)
df = pd.read_csv("./dados/nelson_testeSL.csv", skiprows=[0,1,2,4,5], header=[0,1])
df = df.rename(columns={'Unnamed: 0_level_0': ''})

columns = [
        'Skeleton 016:LFTC', #0
        'Skeleton 016:RIPS', #1
        'Skeleton 016:Head', #2
        'Skeleton 016:LShoulder', #3
        'Skeleton 016:LUArm', #4
        'Skeleton 016:LFLE', #5
        ]

df['Fadiga'] = ['Sim' if x > ((4/5) * len(df)) else 'Nao' for x in df['','Frame']]
df.columns = [f'{i} {j}' for i, j in df.columns]
df2 = df[['Skeleton 016:LFTC X','Skeleton 016:LFTC Y','Skeleton 016:LFTC Z']]

def dados_null(data):
    new_df = []
    for i in data.columns:
        primeiro = True
        sem_na = []
        cont = 0
        nulls = False
        anterior = 0
        for j in data[i]:
            if pd.isna(j) :
                if primeiro:
                    primeiro = False
                    cont = 1
                    nulls = True    
                    continue
                cont += 1
                continue
            if nulls:
                dif = (j - anterior) / (cont + 1)
                for k in range(cont):
                    num = (anterior + ((k+1) * dif))
                    sem_na.append(num)
                
                primeiro = True
                nulls = False
                cont = 0
            sem_na.append(j)
            anterior = j

        if cont != 0:
            for _ in range(cont):
                sem_na.append(anterior)
        new_df.append(sem_na)

    return new_df

new = dados_null(df2)
plt.plot(new[2])
plt.show()
print(np.array(new).T)

new_df = pd.DataFrame(np.array(new).T, columns= ['Skeleton 016:LFTC X','Skeleton 016:LFTC Y','Skeleton 016:LFTC Z'])
print(100 * new_df.isna().sum() / len (new_df))
print(new_df)

new_df.to_csv('nelson.csv',  index=False)