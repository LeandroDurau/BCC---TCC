from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import pandas as pd
import os

from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px

import plotly.graph_objects as go
import plotly.offline


# Incorporate data
#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
df = pd.read_excel("resultados/Teresinha_rf.xlsx")
df = df.sort_values(by="y")

df_tp = df[df['label'] == 'TP']
df_tn = df[df['label'] == 'TN']
df_fp = df[df['label'] == 'FP']
df_fn = df[df['label'] == 'FN']

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_tp['y'], y=df_tp['x'], mode='lines', name=f'TP', line=dict(color="green")))
fig.add_trace(go.Scatter(x=df_tn['y'], y=df_tn['x'], mode='lines', name=f'TN', line=dict(color="royalblue")))
fig.add_trace(go.Scatter(x=df_fp['y'], y=df_fp['x'], mode='lines', name=f'FP', line=dict(color="firebrick")))
fig.add_trace(go.Scatter(x=df_fn['y'], y=df_fn['x'], mode='lines', name=f'FN', line=dict(color="yellow")))

path = './graficos/Teresinha_rf.html'
plotly.offline.plot(fig, filename = path, auto_open=False)
