import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.offline

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

people_skeleton_numbers = {
    # 'katia': 12,
    'nelson': 16,
    'Amaro': 18,
    # 'Adilson': 19,
    # 'Jair': 22,
    'Marilene': 25,
    'Izabel': 27,
    # 'Josui': 30,
    # 'Carlos': 32,
    # 'Arnaldo': 34
}

def generate_columns():
    return [
        # f'Frame',
        f'Z',
        f'Y',
        f'X'
        # f'1_Valley',
        # f'1_Peak'
    ]

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, header=[0], delimiter=";")
    columns = generate_columns()
    return df, columns

########################################################

def calculate_valleys_peaks(df, selectedColumn):
    maximum = df[selectedColumn].max()
    minimum = df[selectedColumn].min()
    globalMean = (maximum + minimum) / 2

    pp = []  # previousPoints
    uh = False  # Uphill
    cv = [[], []]  # cicle valley
    cp = [[], []]  # cicle peak

    for index in range(len(df[selectedColumn])):
        value = df[selectedColumn][index]
        pp.append(value)

        if len(pp) > 30:
            pp.pop(0)
        elif len(pp) < 30:
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

def create_figure(df, columns):
    fig = go.Figure()

    initial_col = columns[0]
    cv, cp = calculate_valleys_peaks(df, initial_col)

    fig.add_trace(go.Scatter(x=df["Frame"], y=df[initial_col], mode='lines', name=f'Eixo'))
    fig.add_trace(go.Scatter(x=df["Frame"].iloc[cv[0]], y=df[initial_col].iloc[cv[0]], mode='markers', name='Vales', marker=dict(color='firebrick', size=10)))
    fig.add_trace(go.Scatter(x=df["Frame"].iloc[cp[0]], y=df[initial_col].iloc[cp[0]], mode='markers', name='Picos', marker=dict(color='royalblue', size=10)))

    buttons = []
    for col in columns:
        cv, cp = calculate_valleys_peaks(df, col)
        button = dict(label=col,
                      method="update",
                      args=[{"y": [df[col], df[col].iloc[cv[0]], df[col].iloc[cp[0]]],
                             "x": [df["Frame"], df["Frame"].iloc[cv[0]], df["Frame"].iloc[cp[0]]]}])
        buttons.append(button)
    fig.update_layout(
        updatemenus=[
            dict(buttons=buttons, direction="down", showactive=True, x=0, xanchor="left", y=1.0, yanchor="top")
    ])
    fig.update_layout(title='LFTC Movement', xaxis_title='Frame', yaxis_title='Position', legend_title='Legend', showlegend=True)

    return fig

def prepareGraph(file_path):
    df, columns = load_and_prepare_data(file_path)
    fig = create_figure(df, columns)
    return fig