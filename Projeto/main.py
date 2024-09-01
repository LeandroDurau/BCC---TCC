from numpy.lib import median
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import xlsxwriter
import math
from scipy.interpolate import UnivariateSpline

id = sys.argv[1]
axis = sys.argv[2]

# Listagem de pessoas
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

# Criação do xlsx
workbook = xlsxwriter.Workbook(f"normalizacao/normalizado-{pessoa}.xlsx")
worksheet = workbook.add_worksheet()

# Leitura de dados
df = pd.read_csv(f"./dados/{pessoa}_testeSL.csv", skiprows=[0,1,2,4,5], header=[0,1])
df = df.rename(columns={'Unnamed: 0_level_0': ''})

# Listagem de pontos uteis
columns = [
    f'Skeleton 0{id}:LFTC', #0
    f"Skeleton 0{id}:RIPS", #1
    f"Skeleton 0{id}:Head", #2
    f"Skeleton 0{id}:LShoulder", #3
    f"Skeleton 0{id}:LUArm", #4
    f"Skeleton 0{id}:LFLE", #5
]

# Seleção do ponto
selectedColumn = columns[0]
selectedColumn2 = columns[0]

# Interpolação linear para remoção de NANs
df[selectedColumn,axis] = df[selectedColumn, axis].interpolate()

max = df[selectedColumn, axis].max()
min = df[selectedColumn, axis].min()
globalMean = (max + min) / 2

pp = [] # Previous points
uh = False # Uphill

cv = [[],[]] #cicle valley
cp = [[],[]] #cicle peak

pv = 0 # Peak ou valley (Para escrita das anotações no excel)

pontos = []
sizes = []

# Variáveis para segmentação
segmentos = []
segmentos_temp = []
last = 0

for index in range(len(df[selectedColumn,axis])):
    value = df[selectedColumn,axis][index]
    pp.append(value)

    # Segmentando o dataset
    segmentos_temp.append(value)

    # Manter pp com 25 pontos
    if len(pp) > 25 :
        pp.pop(0)
    elif len(pp) < 25:
        continue
   
    mean = np.mean(pp)

    #Lógica para achar vales
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
        

    #Lógica para achar picos
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

# Interpolação dos segmentos para terem o mesmo tamanho
for value in segmentos:
    old_indexes = np.arange(0, len(value))
    new_indexes = np.linspace(0, len(value) - 1, int(sizes_median))
    spl = UnivariateSpline(old_indexes, value, k=3, s=0)
    new_val = spl(new_indexes)

    for item in new_val:
        worksheet.write(row, col, item)
        col += 1
    col = 1
    row += 1

workbook.close()
exit()
