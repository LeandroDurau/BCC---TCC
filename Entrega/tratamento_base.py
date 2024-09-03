from numpy.lib import median
import pandas as pd
import numpy as np
import sys
import xlsxwriter
from scipy.interpolate import UnivariateSpline


id = '18'#sys.argv[1]
axis = 'Y'#sys.argv[2]


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

df = pd.read_csv(f"./dados/{pessoa}_testeSL.csv", skiprows=[0,1,2,4,5], header=[0,1])
df = df.rename(columns={'Unnamed: 0_level_0': ''})

dfs = pd.read_excel(f"./dados/{pessoa}.xlsx", sheet_name=f"{axis}")
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
df[selectedColumn,axis] = df[selectedColumn, axis].interpolate()

max = df[selectedColumn, axis].max()
min = df[selectedColumn, axis].min()
globalMean = (max + min) / 2

pp = [] ## previousPoints
uh = False # Uphill

cv = [[],[]] #cicle valley
cp = [[],[]] #cicle peak

helper = 1
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
        helper += 1
        

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

row = 0
col = 1
up_idx=1
down_idx=1

for value in segmentos:
    old_indexes = np.arange(0, len(value))
    new_indexes = np.linspace(0, len(value) - 1, int(sizes_median))
    spl = UnivariateSpline(old_indexes, value, k=3, s=0)
    new_val = spl(new_indexes)

    # Classes
    if value[0] > value[-1]:
        # O ultimo n foi classificado nesse caso
        if (down_idx not in dfs[2]):
            down_idx -= 1
        worksheet.write(row,0, dfs[2][down_idx])
        down_idx += 1
    else:
        if (up_idx not in dfs[2]):
            up_idx -= 1
        worksheet.write(row,0, dfs[0][up_idx])
        up_idx += 1

    for item in new_val:
        worksheet.write(row, col, item)
        col += 1
    col = 1
    row += 1

workbook.close()
