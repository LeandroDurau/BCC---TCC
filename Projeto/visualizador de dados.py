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
df = pd.read_excel("resultados/ilda.xlsx")
df = df.sort_values(by="y")

df_1 = df[df['label'] == 1]
df_0 = df[df['label'] == 0]

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_1['y'], y=df_1['x'], mode='lines', name=f'Eixo', line=dict(color="red")))
fig.add_trace(go.Scatter(x=df_0['y'], y=df_0['x'], mode='lines', name=f'Eixo', line=dict(color="blue")))

path = './graficos/ilda.html'
plotly.offline.plot(fig, filename = path, auto_open=False)
