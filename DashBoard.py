# https://dash.plotly.com/dash-core-components
# https://dash.plotly.com/layout
# https://dash.plotly.com/dash-html-components
# COLOR HUNT PAGE
# Reference Link :
# https://towardsdatascience.com/how-to-use-docker-to-deploy-a-dashboard-app-on-aws-8df5fb322708
# Aknowledges to the autors

#from turtle import width
from dash import Dash, html, dcc  # estas funciones son para el layout 
from dash import Input, Output
import base64

# estas funciones son para el callback de cada uno de las funcionalidades
import plotly.express as px 
import plotly.graph_objects as go
import pandas as pd
import numpy as np

## Librerías del modelo de entrenamiento
from lime import lime_tabular
from sklearn import model_selection

import joblib
seed = 27

app = Dash(__name__)
server = app.server
app.title = "PredictION"
### Con esta sección se crea el dataframe

col=["qubit","depth","reads", "cov"]
dtf = pd.read_csv("data/INS_resum5.1.csv", header=0, delimiter=";",names=col)

### Con esta seccion se grafica con plotly las barras
#fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

### Separación del dataset
global X_test
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.25,random_state=seed)
X_names = ['qubit','depth','cov']
X_train = dtf_train[X_names].values
y_train = dtf_train["reads"].values
X_test = dtf_test[X_names].values
y_test = dtf_test["reads"].values

global model, explainer, error
#model = ensemble.GradientBoostingRegressor()

model = joblib.load('models/finalized_model.pkl') # Carga del modelo.
#model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predicciones = model.predict(dtf[X_names].values)
#error = round(np.mean(np.abs((y_test-y_pred)/y_pred)), 2)
error = round((1/len(y_pred))*sum(np.abs(y_test-y_pred)))

df = pd.DataFrame()
df = dtf
df['predic'] = predicciones
#print(X_test[0])
fig1 = go.Figure()
fig1.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    },title='Local explanation',transition_duration=500,width=600, height=400)
fig1.update_xaxes(mirror=False,zeroline=True, zerolinewidth=2, zerolinecolor='lightslategrey')
fig1.update_yaxes(mirror=False,zeroline=True, zerolinewidth=2, zerolinecolor='lightslategrey')

fig2 = px.scatter(df, x="reads", y="predic", trendline="ols",width=600, height=400)
fig2.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    },title='Regression Model',transition_duration=500,width=600, height=400)
fig2.update_xaxes( mirror=True,gridwidth=1, gridcolor='lightslategrey',zeroline=True, zerolinewidth=2, zerolinecolor='lightslategrey')
fig2.update_yaxes(mirror=True,zeroline=True, zerolinewidth=2, zerolinecolor='lightslategrey')
explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names="Y", mode="regression",random_state=7)
figtemp = fig2
### Acá agregamos el enlace de la imagen para el logo

image_filename = 'images/univalle.jpg' # replace with your own image
image_group = 'images/neas.jpg'
image_people = 'images/peoplecontact.png'
image_unal = 'images/logoUNAL.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
encoded2_image = base64.b64encode(open(image_group, 'rb').read())
encoded3_image = base64.b64encode(open(image_unal, 'rb').read())
encoded4_image = base64.b64encode(open(image_people, 'rb').read())
### vamos a definir el layout


app.layout = html.Div(children=[

    ## Encabezado con |Logo|titulo|autores(opcional)|
    html.Div(children=[
                        html.Div(children=[
                                        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                                                 height='40%', width='40%')
                                          ],style={'float':'left', 'width':'15%','text-align': 'left'}),
                        html.Div(children=[
                                        html.Img(src='data:image/png;base64,{}'.format(encoded3_image.decode()),
                                                 height='60%', width='60%')
                                          ],style={'float':'left', 'width':'15%','text-align': 'left'}),
                        html.Div(children=[
                                        html.H2(children='PredictION: A predictive model to establish performance of Oxford sequencing reads of SARS-CoV-2',style={'text-align': 'center'})
                                          ],style={'float':'left', 'width':'40%'}),
                        
                        html.Div(children=[
                                        html.Img(src='data:image/png;base64,{}'.format(encoded2_image.decode()),
                                                 height='60%', width='60%')
                                          ],style={'float':'left', 'width':'15%','text-align': 'right'}),
                        html.Div(children=[
                                        html.Img(src='data:image/png;base64,{}'.format(encoded4_image.decode()),
                                                 height='90%', width='90%')
                                          ],style={'float':'left', 'width':'15%','text-align': 'right'})
                      ],style={'float':'none','width':'100%'}),
    ## Colocamos algunos espacios
    html.Br(),
    html.Br(),
    html.Br(),
    ## Esta es la descripción del tablero o si se quiere puede ser el instructivo o texto informativo
    html.Div(children=['''
        With PredictION you may estimate sequence reads given a priori  known concentration of cDNA (ng/µl) per sample and the desired coverage depth (mean) and coverage per genome (percentage)
    ''',html.P()],style={'float':'left', 'width':'100%'}),
    
    
    ## Acá está la seccion de los inputs y salidas numericas del tablero |Sliders|inputTexts|SalidasModelo
    html.Div(children=[
        html.Div(children=[
            html.Label('Concentration of cDNA (ng/µl)'),
            dcc.Slider(
                min=round(min(dtf['qubit'].astype(float))), # Minimo rango del slider
                max=round(max(dtf['qubit'].astype(float))), # Máximo rango del slider
                #marks={i: f'Label {i}' if i == 8 else str(i) for i in range(1, 9)},
                #marks={i: str(i) for i in range(0, max(dtf['qubit'].astype(int)),4)}, # Marks son las xticks que va a tener el slider
                value=X_test[1][1], # Este es el valor inicial
                id = 'qubit' 
                ),
            html.Label('Coverage depth (mean)'),
            dcc.Slider(
                min=round(min(dtf['depth'].astype(float))), # Minimo rango del slider
                max=round(max(dtf['depth'].astype(float))), # Máximo rango del slider
                #marks={i: f'Label {i}' if i == 8 else str(i) for i in range(1, 9)},
                #marks={i: f'{i}k' for i in range(0, max(dtf['reads'].astype(int)),1000)}, # Marks son las xticks que va a tener el slider
                value=X_test[1][0], # Este es el valor inicial
                id = 'depth' 
                ),
            html.Label('Coverage per genome (percentage)'),
            dcc.Slider(
                min=round(min(dtf['cov'].astype(float))), # Minimo rango del slider
                max=round(max(dtf['cov'].astype(float))), # Máximo rango del slider
                #marks={i: f'Label {i}' if i == 8 else str(i) for i in range(1, 9)},
                #marks={i: str(i) for i in range(0, max(dtf['cov'].astype(int)),10)}, # Marks son las xticks que va a tener el slider
                value=X_test[1][2], # Este es el valor inicial
                id = 'cov' 
                )
        ],style={'float':'left', 'width':'33%'}), # style={'float':'left', 'width':'50%','display':'block'}),
    
        html.Div(children=[
            html.Br(),
            html.Div([
            "Concentration of cDNA (ng/µl):",
            dcc.Input(id='Qubitbox', value=round(X_test[1][1]), type='number') # Este es el bloque de entrada para pedirle un número al usuario
            ],style={'text-align': 'right'}),
            
            html.Br(), html.Br(),
            html.Div([
            "Coverage depth (mean):",
            dcc.Input(id='Depthbox', value=round(X_test[1][0]), type='number') # Este es el bloque de entrada para pedirle un número al usuario
            ],style={'text-align': 'right'}),
            html.Br(), html.Br(),
            html.Div([
            "Coverage per genome (percentage):",
            dcc.Input(id='Coverage', value=round(X_test[1][2]), type='number') # Este es el bloque de entrada para pedirle un número al usuario
            ],style={'text-align': 'right'}),
            html.Br()
            ],style={'float':'left', 'width':'33%'}), # Colocamoss un texto que dice mi salida

        html.Div(children=[
            html.Br(),
            html.Br(), html.Br(),
            html.Div(id='my-output', style={'font-size': '150%','text-align': 'center'}), # Colocamoss un texto que dice mi salida
            html.Br(), 
            html.Div(id='my-output2', style={'font-size': '150%','text-align': 'center'}), # Colocamoss un texto que dice mi salida     
            #html.Div(id='my-output2', style={'font-size': '150%','text-align': 'center'}),      
            html.Br()],style={'float':'left', 'width':'33%'})

        ],style={'float':'left', 'width':'100%'}),

    html.Br(),
    html.Br(),  
    ## Este es el bloque de las gráficas |localExplanation|regressionPredicted|
    html.Div(children=[
        html.Div(children=[dcc.Graph(id='graph',figure=fig1)],style={'float':'left', 'width':'45%'}),
        html.Div(children=[html.Br()],style={'float':'left', 'width':'10%'}),
        html.Br(),
        #html.Br(),
        html.Div(children=[dcc.Graph(id='graph2', figure=fig2)],style={'float':'right', 'width':'45%'})
    ],style={'float':'left', 'width':'100%',}),   

    html.Div(children=[
        html.H3('Authors: '),
        #html.H4('David E. Valencia-Valencia <sup>1§<\sup>, Diana López-Alvarez1,2,3,§, Nelson Rivera Franco1,2, Andres Castillo1, Johan S. Piña4, and Beatriz Parra2'),
        dcc.Markdown('''David E. Valencia-Valencia$$^{1§}$$, Diana López-Alvarez$$^{1,2,3,§}$$, Nelson Rivera Franco$$^{1,2}$$, Andres Castillo$$^1$$, Johan S. Piña$$^4$$, and Beatriz Parra$$^2$$''', mathjax=True),
        html.Br(),
        dcc.Markdown('$$^1$$ Laboratorio de Técnicas y Análisis Ómicos - TAOLab/CiBioFi, Facultad de Ciencias Naturales y Exactas, Universidad del Valle, Cali, Colombia', mathjax=True),
        dcc.Markdown('$$^2$$ Grupo VIREM - Virus Emergentes y Enfermedad, Escuela de Ciencias Básicas, Facultad de Salud, Universidad del Valle, Cali, Colombia', mathjax=True),
        dcc.Markdown('$$^3$$ Departamento de Ciencias Biológicas, Facultad de Ciencias Agropecuarias, Universidad Nacional de Colombia, Palmira, Colombia', mathjax=True),
        dcc.Markdown('$$^4$$ Department of Data Science, People Contact, Manizales, Caldas, Colombia', mathjax=True),
        dcc.Markdown('$$^§$$ Both authors contributed equally to this work.', mathjax=True),
        html.H3('Acknowledgements: '),
        dcc.Markdown('We are thankful to all VIREM team (“Virus Emergentes y Enfermedades”), who are supporting the diagnostics of SARS-CoV-2 in Colombia, as well as different investigators of Colombia and worldwide who deposited their genomes in GISAID. We also thank the Colombian network for SARS-CoV-2 genomic surveillance led by the National Institute of Health (INS). Thanks to CEO Diego Ceballos of People Contact enterprise for supporting in dashboard builder. The study has been funded by NIH–Fogarty Grant number R01NS110122.'),
    ],style={'float':'left', 'width':'100%'})
    ]) ## Cerramos el children más grande de la página HTML


@app.callback(
    Output(component_id='my-output',component_property='children'),
    Output(component_id='my-output2',component_property='children'),
    Output(component_id='graph',component_property='figure'),
    Output(component_id='graph2',component_property='figure'),
    Input(component_id='qubit',component_property='value'),
    Input(component_id='depth',component_property='value'),
    Input(component_id='cov',component_property='value'),
    Input(component_id='graph2',component_property='figure'),
)
def update_graph(qubit_val,depth_val,cov_val,figtemp):
    To_predic = [qubit_val, depth_val, cov_val]
    pred = model.predict([To_predic])[0]
    Text_output = 'Model Prediction: {:,.0f} reads'.format(pred)
    error_output = 'Mean Absolute error (MAE): ± {:,.0f} reads'.format(error)
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
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    },title='Local explanation',transition_duration=500,width=600, height=400)
    fig.update_xaxes( mirror=True,gridwidth=1, gridcolor='lightslategrey',zeroline=True, zerolinewidth=2, zerolinecolor='lightslategrey')
    #fig.update_yaxes( zeroline=True, zerolinewidth=2, zerolinecolor='lightslategrey')


    fig2 = go.Figure(data=figtemp)
    fig2.add_trace(go.Scatter(x=[pred],y=[pred],marker_symbol='x',marker_size=15,name='Predicted'))
    fig2.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    },title='RegressionModel',transition_duration=500,width=600, height=400)

    return Text_output, error_output, fig, fig2


@app.callback(
    Output(component_id='qubit',component_property='value'),
    Output(component_id='depth',component_property='value'),
    Output(component_id='cov',component_property='value'),
    Input(component_id='Qubitbox',component_property='value'),
    Input(component_id='Depthbox',component_property='value'),
    Input(component_id='Coverage',component_property='value')
    )
def sliderChanges(Qubitbox,Depthbox,Coverage):
    
    if(Qubitbox is None):
        Qubitbox = 0
    if(Depthbox is None):
        Depthbox = 0  
    if(Coverage is None):
        Coverage = 0 
    ## Acá va elk codigo opara que cuando cambi el slider entonces se cambien los valores de los cuadros de texto input
    return float(Qubitbox),float(Depthbox),float(Coverage)


@app.callback(
    Output(component_id='Qubitbox',component_property='value'),
    Output(component_id='Depthbox',component_property='value'),
    Output(component_id='Coverage',component_property='value'),
    Input(component_id='qubit',component_property='value'),
    Input(component_id='depth',component_property='value'),
    Input(component_id='cov',component_property='value'),
    )
def boxesChanges(Qubit,Depth,Cov):
    
    if(Qubit is None):
        Qubit = 0
    if(Depth is None):
        Depth = 0  
    if(Cov is None):
        Cov = 0
    ## Acá va el codigo opara que cuando cambi el slider entonces se cambien los valores de los cuadros de texto input
    return float(Qubit),float(Depth),float(Cov)


if __name__ == '__main__':
    app.run_server(debug=False)
