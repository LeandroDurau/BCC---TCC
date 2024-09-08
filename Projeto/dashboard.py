import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import scipy
import plotly.graph_objects as go
import plotly.offline
import sys

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

print(plotly.__version__)

lista_pessoas = {
    '12': 'katia',
    '16': 'nelson',
    '18': 'Amaro',
    '19': 'Adilson',
    '22': 'Jair',
    '25': 'Marilene',
    '27': 'Izabel',
    '30': 'Josui',
    '32': 'Carlos',
    '34': 'Arnaldo',
    '44': 'ilda',
    '23': 'Iria',
    '26': 'Elisabete',
    '09': 'noemi',
    '42': 'Josefina',
    '33': 'Reinaldo',
    '35': 'Pedro',
    '17': 'Valdir',
    '14': 'mariajose',
    '08': 'teresinha',
}


def generate_columns(person_id):
    return [
        f'Skeleton 0{person_id}:LFTC',
        f'Skeleton 0{person_id}:RIPS',
        f'Skeleton 0{person_id}:Head',
        f'Skeleton 0{person_id}:LShoulder',
        f'Skeleton 0{person_id}:LUArm',
        f'Skeleton 0{person_id}:LFLE'
    ]

def load_and_prepare_data(person_id):
    pessoa = lista_pessoas[person_id]
    file_path = "./dados/" + pessoa + "_testeSL.csv"
    df = pd.read_csv(file_path, delimiter=",", skiprows=[0,1,2,4,5], header=[0,1])
    df = df.rename(columns={"Unnamed: 0_level_0": ""})
    return df, generate_columns(person_id)

def calculate_valleys_peaks(df, selectedColumn, axis):
    maximum = df[selectedColumn, axis].max()
    minimum = df[selectedColumn, axis].min()
    globalMean = (maximum + minimum) / 2

    pp = []  # previousPoints
    uh = False  # Uphill
    cv = [[], []]  # cicle valley
    cp = [[], []]  # cicle peak

    for index in range(len(df[selectedColumn, axis])):
        value = df[selectedColumn, axis][index]
        pp.append(value)

        if len(pp) > 25:
            pp.pop(0)
        elif len(pp) < 25:
            continue

        mean = np.mean(pp)
        if mean < value and not uh and value < globalMean:
            uh = True
            cv[0].append(index)
            cv[1].append(value)

        if mean > value and uh and value > globalMean:
            uh = False
            cp[0].append(index)
            cp[1].append(value)

    return cv, cp

def create_figure(df, columns, axis, person_id):
    fig = go.Figure()

    initial_col = columns[0]
    cv, cp = calculate_valleys_peaks(df, initial_col, axis)
    
    numbers1 = []
    numbers2 = []

    for i in range(len(cv[0])):
        numbers1.append(i+1)

    for i in range(len(cp[0])):
        numbers2.append(i+1)
    
    fig.add_trace(go.Scatter(x=df["", "Frame"], y=df[initial_col, axis], mode='lines', name=f'Eixo'))
    fig.add_trace(go.Scatter(x=df["", "Frame"].iloc[cv[0]], y=df[initial_col, axis].iloc[cv[0]], mode='markers+text', text=numbers1, textposition='bottom center', name='Vales', marker=dict(color='firebrick', size=10)))
    fig.add_trace(go.Scatter(x=df["", "Frame"].iloc[cp[0]], y=df[initial_col, axis].iloc[cp[0]], mode='markers+text', text=numbers2, textposition='top center', name='Picos', marker=dict(color='royalblue', size=10)))

    buttons = []
    eixos = ["Y", "X", "Z"]
    for eixo in eixos:
        if eixo == "Z":
            cp, cv = cv, cp

        button = dict(label=eixo,
            method="update",
            args=[{
            "y": [df[initial_col, eixo], df[initial_col, eixo].iloc[cv[0]], df[initial_col, eixo].iloc[cp[0]]],
            "x": [df["", "Frame"], df["", "Frame"].iloc[cv[0]], df["", "Frame"].iloc[cp[0]]], 
            }
        ])

        buttons.append(button)
    fig.update_layout(
        updatemenus=[
            dict(buttons=buttons, direction="down", showactive=True, x=0, xanchor="left", y=1.0, yanchor="top")
    ])
    fig.update_layout(title=lista_pessoas[person_id] + ': LFTC Movement', xaxis_title='Frame', yaxis_title='Position', legend_title='Legend', showlegend=True)

    return fig

def prepareGraph(person_id, axis):
    df, columns = load_and_prepare_data(person_id)
    fig = create_figure(df, columns, axis, person_id)
    return fig

# Pessoa Analisada
# id = sys.argv[1]
axis = "Y"

for id in lista_pessoas.keys():
    fig = prepareGraph(id, axis)
    path = './graficos/'+lista_pessoas[id] + '.html'
    plotly.offline.plot(fig, filename = path, auto_open=False)