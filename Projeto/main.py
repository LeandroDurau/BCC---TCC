import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import xlsxwriter


id = sys.argv[1]
axis = sys.argv[2]


lista_pessoas = {
    '08': 'Teresinha',
    '09': 'Noemi',
    '12': 'Katia',
    '14': 'mariajose',
    '16': 'Nelson',
    '17': 'Valdir',
    '18': 'Amaro',
    '19': 'Adilson',
    '22': 'Jair',
    '23': 'Iria',
    '25': 'Marilene',
    '26': 'Elisabete',
    '27': 'Izabel',
    '30': 'Josui',
    '32': 'Carlos',
    '33': 'Reinaldo',
    '34': 'Arnaldo',
    '35': 'Pedro',
    '42': 'Josefina',
    '44': 'ilda',
}

pessoa = lista_pessoas[id]

# workbook = xlsxwriter.Workbook(f"pontos/pontos-{pessoa}.xlsx")
# worksheet = workbook.add_worksheet()

df = pd.read_csv(f"./dados/{pessoa}_testeSL.csv", skiprows=[0,1,2,4,5], header=[0,1], nrows=14400)
df = df.rename(columns={'Unnamed: 0_level_0': ''})

columns = [
    f'Skeleton 0{id}:LFTC', #0
    f"Skeleton 0{id}:RIPS", #1
    f"Skeleton 0{id}:Head", #2
    f"Skeleton 0{id}:LShoulder", #3
    f"Skeleton 0{id}:LUArm", #4
    f"Skeleton 0{id}:LFLE", #5
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


pontos = []

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
        pontos.append(df['', 'Frame'][index])

diferencas = []

for index in range(len(pontos)):
    if index == 0:
        continue
    diferencas.append(pontos[index] - pontos[index - 1])

media = np.mean(diferencas)
print(f'{pessoa} - {media} frames - {media/120}s')
# worksheet.write('A1', 'Tempo')
# worksheet.write('A2', 'Pontos')
# worksheet.write('B1', tempos)
# worksheet.write('B2', pontos)
#
#
# workbook.close()
# fig,ax = plt.subplots()
# sns.lineplot(y=df[selectedColumn2,axis], x = df["","Frame"])
# ax.scatter(cv[0], cv[1], c='firebrick', zorder=100)
# ax.scatter(cp[0], cp[1], c='orange', zorder=100)
#
# for i in range(len(cv[0])):
#     ax.annotate(str(i+1), (cp[0][i], cp[1][i] + 1))
#     ax.annotate(str(i+1), (cv[0][i], cv[1][i] - 10))
#
# plt.show()
