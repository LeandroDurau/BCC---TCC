from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import pandas as pd
import os

'''
df_total = pd.DataFrame()
for i in os.listdir('normalizacao/'):
    df = pd.read_excel(f"normalizacao/{i}",header=None)
    df['Nome'] = i.split('-')[1].split('.')[0]
    df_total = pd.concat([df,df_total])
df_total = df_total.rename(columns={1:'Qtd Total Pontos',2:'Qtd Dados'})


np_x = df_total.drop([0,'Nome','Qtd Dados','Qtd Total Pontos'],axis = 1).to_numpy()
pessoas = df_total['Nome'].to_numpy()
classes = df_total[0].to_numpy()
plot_x = []
nome = ''
count = -1
for line in range(len(np_x)):
    if nome != pessoas[line]:
        count += 1
        plot_x.append([[],[],[]])
        plot_x[count][0] = pessoas[line]
        plot_x[count][1] = [[],[]]
        plot_x[count][2] = [[],[]]
        nome = pessoas[line]
        y = 0
    for x in np_x[line]:
        if classes[line] == 'Normal':
            plot_x[count][1][0].append(x)
            plot_x[count][1][1].append(y)
        else:
            plot_x[count][2][0].append(x)
            plot_x[count][2][1].append(y)
        y += 1
        '''


from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px

# Incorporate data
#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
df = pd.read_excel("resultados/ilda.xlsx")
df = df.sort_values(by="y")
print(df.head(5))

# Initialize the app
app = Dash()
from base64 import b64encode
import io
buffer = io.StringIO()
html_bytes = buffer.getvalue().encode()
encoded = b64encode(html_bytes).decode()
# App layout
app.layout = (
    html.Div(
        [
            html.Div(children='My First App with Data and a Graph'),
            dcc.Graph(figure=px.line(df, x='y', y='x',color='label')),
            html.A(
                html.Button("Download as HTML"), 
                id="download"
            ),
            dcc.Download(id='download_1')

        ]
    )
)
@app.callback(
    Output('download_1','data'),
    Input('download','n_clicks'),prevent_initial_call=True)
def download_html(n):
    return dcc.send_file("plotly_graph.html")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)