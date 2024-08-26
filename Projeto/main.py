from numpy.lib import median
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import xlsxwriter
import math


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

workbook = xlsxwriter.Workbook(f"normalizacao/normalizado-{pessoa}.xlsx")
worksheet = workbook.add_worksheet()

# workbook = xlsxwriter.Workbook(f"./anotacoes_realizadas/reestrutura_{pessoa}_{axis}.xlsx", {'nan_inf_to_errors': True})
# worksheet = workbook.add_worksheet(axis)

df = pd.read_csv(f"./dados/{pessoa}_testeSL.csv", skiprows=[0,1,2,4,5], header=[0,1])
df = df.rename(columns={'Unnamed: 0_level_0': ''})

dfs = pd.read_excel(f"./anotacoes_realizadas/{pessoa}.xlsx", sheet_name=f"{axis}")
dfs = dfs.transpose()


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

# helper = 1
# helper2 = 1
pv = 0

pontos = []
sizes = []

segmentos = []
segmentos_temp = []
last = 0

for index in range(len(df[selectedColumn,axis])):
    value = df[selectedColumn,axis][index]
    pp.append(value)

    # Segmentando o dataset
    segmentos_temp.append(value)

    # val = dfs[pv][helper]
    # try:
    #     worksheet.write(f'A{index}', val)
    # except:
    #     pass

    if len(pp) > 25 :
        pp.pop(0)
    elif len(pp) < 25:
        continue
   
    mean = np.mean(pp)
    if (mean < value and not uh and value < globalMean):
        # Muda a direção
        uh = True

        # acha o tamanho do range
        sizes.append(index-last)
        last = index

        # Salva o ponto
        cv[0].append(index)
        cv[1].append(df[selectedColumn2,axis][index])

        # segmentação
        segmentos.append(segmentos_temp.copy())
        segmentos_temp = []
        pv = 2
        # helper += 1
        

    if (mean > value and uh and value > globalMean):
        # Muda a direção
        uh = False

        # acha o tamanho do range
        sizes.append(index-last)
        last = index

        # Salva o ponto
        cp[0].append(index)
        cp[1].append(df[selectedColumn2,axis][index])
        pontos.append(df['', 'Frame'][index])

        # segmentação
        segmentos.append(segmentos_temp.copy())
        segmentos_temp = []

        pv = 0

sizes.sort()
sizes_median = median(sizes)

print(sizes_median)

row = 0
col = 1
up_idx=1
down_idx=1

for value in segmentos:
    if (len(value) > sizes_median):
        dif = int(len(value) - sizes_median)
        aCada = len(value) // dif

        for x in range(dif):
            value.pop(x * aCada - x)

    elif (len(value) < sizes_median):
        dif = int(sizes_median - len(value))
        aCada = len(value) // dif

        for x in range(dif):
            index = x * aCada - x;
            if index == 0: 
                index = 1
            value.insert(index, value[index-1])

    # Classes
    if value[0] > value[-1]:
        worksheet.write(row,0, dfs[2][down_idx])
        down_idx += 1
    else:
        worksheet.write(row,0, dfs[0][up_idx])
        up_idx += 1

    #Passa pelo segmento a adiciona cada item a uma coluna da linha
    for item in value:
        if (math.isnan(item)):
            item = value[col-2]
            value[col] = value[col-2]
        print(value[col-2], item)
        worksheet.write(row, col, item)
        col += 1
    col = 1
    row += 1


workbook.close()
exit()

# diferencas = []

# for index in range(len(pontos)):
#     if index == 0:
#         continue
#     diferencas.append(pontos[index] - pontos[index - 1])

# media = np.mean(diferencas)
# print(f'{pessoa} - {media} frames - {media/120}s')
# worksheet.write('A1', 'Tempo')
# worksheet.write('A2', 'Pontos')
# worksheet.write('B1', tempos)
# worksheet.write('B2', pontos)
#
#
# workbook.close()
fig,ax = plt.subplots()
sns.lineplot(y=df[selectedColumn2,axis], x = df["","Frame"])
ax.scatter(cv[0], cv[1], c='firebrick', zorder=100)
ax.scatter(cp[0], cp[1], c='orange', zorder=100)
#
for i in range(len(cv[0])):
    ax.annotate(str(i+1), (cp[0][i], cp[1][i] + 1))
    ax.annotate(str(i+1), (cv[0][i], cv[1][i] - 10))

plt.show()
