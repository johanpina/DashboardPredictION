# https://dash.plotly.com/dash-core-components
# https://dash.plotly.com/layout
# https://dash.plotly.com/dash-html-components

# Reference Link :
# https://towardsdatascience.com/how-to-use-docker-to-deploy-a-dashboard-app-on-aws-8df5fb322708
# Aknowledges to the autors


from turtle import width
from dash import Dash, html, dcc  # estas funciones son para el layout 
from dash import Input, Output

# estas funciones son para el callback de cada uno de las funcionalidades
import plotly.express as px 
import plotly.graph_objects as go
import pandas as pd
import numpy as np

## Librerías del modelo de entrenamiento
from lime import lime_tabular
from sklearn import model_selection, ensemble
import joblib

app = Dash(__name__)


### Con esta sección se crea el dataframe
'''df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})'''

col=["qubit","depth","reads", "cov"]
dtf = pd.read_csv("INS_resum_end.csv", header=0, delimiter=";")

### Con esta seccion se grafica con plotly las barras
#fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

### Separación del dataset
global X_test
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3,random_state=7)
X_names = ['depth', 'qubit', 'cov']
X_train = dtf_train[X_names].values
y_train = dtf_train["reads"].values
X_test = dtf_test[X_names].values
y_test = dtf_test["reads"].values

global model, explainer
#model = ensemble.GradientBoostingRegressor()
model = joblib.load('ModelTrained.pkl') # Carga del modelo.
#model.fit(X_train, y_train)

#print(X_test[0])

explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names="Y", mode="regression",random_state=7)


### vamos a definir el layout


app.layout = html.Div(children=[
    html.H1(children='Tablero Dinámico'),  # Coloca el Texto Hola Dash

    html.Div(children='''
        Dinamic Board For Reads Regression: This applications use regression to predict the numbers of reads of a secuenciation process.
    '''),
    html.Br(),

    html.Div(children=[html.Label('Qubit:'),
        dcc.Slider(
            min=min(dtf['qubit'].astype(float)), # Minimo rango del slider
            max=max(dtf['qubit'].astype(float)), # Máximo rango del slider
            #marks={i: f'Label {i}' if i == 8 else str(i) for i in range(1, 9)},
            #marks={i: str(i) for i in range(0, max(dtf['qubit'].astype(int)),4)}, # Marks son las xticks que va a tener el slider
            value=X_test[1][1], # Este es el valor inicial
            id = 'qubit' 
        ),
        html.Label('Depth'),
        dcc.Slider(
            min=min(dtf['depth'].astype(float)), # Minimo rango del slider
            max=max(dtf['depth'].astype(float)), # Máximo rango del slider
            #marks={i: f'Label {i}' if i == 8 else str(i) for i in range(1, 9)},
            #marks={i: f'{i}k' for i in range(0, max(dtf['reads'].astype(int)),1000)}, # Marks son las xticks que va a tener el slider
            value=X_test[1][0], # Este es el valor inicial
            id = 'depth' 
        ),
        html.Label('Coverage'),
        dcc.Slider(
            min=min(dtf['cov'].astype(float)), # Minimo rango del slider
            max=max(dtf['cov'].astype(float)), # Máximo rango del slider
            #marks={i: f'Label {i}' if i == 8 else str(i) for i in range(1, 9)},
            #marks={i: str(i) for i in range(0, max(dtf['cov'].astype(int)),10)}, # Marks son las xticks que va a tener el slider
            value=X_test[1][2], # Este es el valor inicial
            id = 'cov' 
        )],style={'float':'left', 'width':'50%'}), # style={'float':'left', 'width':'50%','display':'block'}),
    
    html.Div(children=[
        html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='initial value', type='text') # Este es el bloque de entrada para pedirle un número al usuario
        ]),
        html.Br(),
        html.Div(id='my-output', style={'font-size': '150%'}), # Colocamoss un texto que dice mi salida
        html.Br(),
        html.Div(id='my-output2', style={'font-size': '150%'})],style={'float':'left', 'width':'50%'}), # Colocamoss un texto que dice mi salida
    html.Div(children=[dcc.Graph(
        id='graph'
    )],style={'float':'left', 'width':'100%'})
    
    ]) ## Cerramos el children más grande de la página HTML

@app.callback( # este callback se declara para obtener datos de los componentes y poder realizar funcionalidades con ellos
    Output(component_id='my-output', component_property='children'), # La salida de la funcion que se defina abajo sale a este elemento
    Input(component_id='my-input', component_property='value') # La entrada de la funcion declarada abajo es la entrada de la función.
)
def update_output_div(input_value):
    return f'Predicción del Modelo: {input_value}'

@app.callback(
    Output(component_id='my-output2',component_property='children'),
    Output(component_id='graph',component_property='figure'),
    Input(component_id='qubit',component_property='value'),
    Input(component_id='depth',component_property='value'),
    Input(component_id='cov',component_property='value')
)
def update_graph(qubit_val,reads_val,cov_val):
    To_predic = [qubit_val, reads_val, cov_val]
    pred = model.predict([To_predic])[0]
    Text_output = 'Output: {:,.0f}'.format(pred)
    explained = explainer.explain_instance(np.asarray(To_predic), model.predict, num_features=10)
    salida = explained.as_list()

    vals = [x[1] for x in salida]
    names = [x[0] for x in salida]
    vals.reverse()
    names.reverse()
    colors = ['rgba(50, 171, 96, 0.6)' if x > 0 else 'rgba(255, 15, 15, 0.6)' for x in vals]
    lines = ['rgba(50, 171, 96, 1.0)' if x > 0 else 'rgba(255, 15, 15, 1.0)' for x in vals]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=vals,
        name='Predic',
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color=lines, width=3)
        )
    ))
    fig.update_layout(title='Local explanation',transition_duration=500)

    return Text_output, fig



if __name__ == '__main__':
    app.run_server(host = '0.0.0.0', port=8050, debug=True)
