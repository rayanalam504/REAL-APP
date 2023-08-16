
import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import plotly.io as pio
import json
# import dash_bootstrap_components as dbc


def compute(initial_data, mass):

    n = len(initial_data[0])

    def ODE(t,s):
        dsdt = []
        for i in range(n):
            ax, ay = 0, 0
            x,y = s[i*4],s[i*4+1]
            for j in range(n):
                if j==i:
                    continue
                xj,yj = s[j*4],s[j*4 + 1]
                r = np.sqrt((xj - x)**2 + (yj - y)**2)
                ax += mass[j]/r**3 * (xj-x)
                ay += mass[j]/r**3 * (yj-y)

            vx, vy = s[i*4 + 2],s[i*4 + 3]
            dsdt += vx, vy, ax, ay
        return dsdt

    s0=[]
    for i in range(n):
        for j in range(len(initial_data)):
            s0 += [initial_data[j][i]]

    t_start = 0
    t_end = 60
    steps = 2000
    t = np.linspace(t_start, t_end, steps)

    data = solve_ivp(ODE, [t_start, t_end], s0, method='DOP853', t_eval=t, rtol=1e-10, atol=1e-13)

    x,y = [],[]

    for i in range(0,len(data.y),4):
        x.append(data.y[i])
        y.append(data.y[i+1])

    return x,y



input_style = {"width": "60px", 'background-color': '#141f24', 'color': 'aliceblue'}


# Global variable to store the data
bodies_data = []
layout = html.Div(
  children=[
    html.Div(
      children=[
        html.H1('The n-body Problem:'),
        html.Label('On this app you can create visualizations for the n-body problem.'),
        html.Br(),
        html.Label('To get started, pick a preset or create your own.'),
        html.Br(),
        html.Label('Once you have finished creating initial conditions, click submit to render.'),
        html.Br(),
        html.Br(),
        html.Label('Presets (set number of bodies to 3):'),
        html.Div(id='infinity-preset', children=[html.Button('Infinity', id='preset-button')]),
        html.Div(id='yin-yang', children=[html.Button('Yin Yang', id='unequal-mass-preset-button')]),
        html.Div(id='free-fall', children=[html.Button('Free Fall', id='free-fall-preset-button')]),
        html.Div(id='nan', children=[html.Button('Unequal Masses', id='unequal-mass')]),
        html.Br(),
        html.Br(),
        html.Div(id='bodies-input', children=[html.Label('Number of bodies:'),
        html.Br(),
        dcc.Input(id='num-bodies', type='number', value=3, min=1, style=input_style),]),
        html.Div(id='bodies-container', 
        children=[
            html.Div(id='bodies-inputs'),
            html.Button('Submit', id='submit-button')
                ]
            ),
        html.Div(id='confirmation-message'),
        #html.Button('Next', id='next-button'),
        dcc.Loading(id='loading',type='default', color='aliceblue', children=[
                html.Div(id='express-plot-container')
                ], 
            ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        dcc.Graph(id='scatter-plot', style={"width": "30%", "position": "absolute", "top": "200px", "right": "10px"})
                            ]
                        )
                     ]
                 )
               ]
            )
        ],style={'display': 'flex', 'flex-direction': 'row', "background-color": "#111111"}
    )

######################################################################################################################
######################################################################################################################
@callback(
    [
        Output({'type': 'x', 'index': dash.dependencies.ALL}, 'value'),
        Output({'type': 'y', 'index': dash.dependencies.ALL}, 'value'),
        Output({'type': 'vx', 'index': dash.dependencies.ALL}, 'value'),
        Output({'type': 'vy', 'index': dash.dependencies.ALL}, 'value'),
        Output({'type': 'mass', 'index': dash.dependencies.ALL}, 'value')
    ],
    [
        Input('preset-button', 'n_clicks'),
        Input('unequal-mass-preset-button', 'n_clicks'),
        Input('free-fall-preset-button', 'n_clicks'),
        Input('unequal-mass', 'n_clicks'),
    ],
    prevent_initial_call=True
)

def load_preset_data(infinity_clicks, yinyang_clicks, freefall_clicks, unequal_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "preset-button":
        data = [
            [-0.97000436, 0.97000436, 0.0],
            [0.24308753, -0.24308753, 0.0],
            [0.466203685, 0.466203685, -0.93240737],
            [0.43236573, 0.43236573, -0.86473146]
        ]
        mass = [1,1,1]
    elif button_id == "unequal-mass-preset-button":
        data = [
            [-1, 1, 0],
            [0, 0, 0],
            [0.513938, 0.513938, -1.027876],
            [0.304736, 0.304736, -0.609472]
        ]
        mass = [1,1,1]
    elif button_id == "free-fall-preset-button":
        data = [
            [-0.5, 0.5, 0.0207067154],
            [0, 0, 0.3133550361],
            [0, 0, 0],
            [0, 0, 0]
        ]
        mass = [1,1,1]
    elif button_id == "unequal-mass":
        data = [
            [-1, 1, 0],  
            [0, 0, 0],  
            [0.2374365149, 0.2374365149, -0.9497460596],  
            [0.2536896353, 0.2536896353, -1.0147585412]  
        ]
        mass = [1,1,0.5]
    else:
        raise PreventUpdate

    return data[0], data[1], data[2], data[3], mass

######################################################################################################################
######################################################################################################################

######################################################################################################################
@callback(
    Output('bodies-inputs', 'children'),
    Input('num-bodies', 'value')
)
def create_body_inputs(num_bodies):
    inputs = []
    input_style = {"width": "60px", 'background-color': '#141f24', 'color': 'aliceblue'}
    label_style = {"color": "white", "display": "inline-block", "margin-right": "10px"}

    for i in range(num_bodies):
        inputs.append(
            html.Div([
                html.Label(f'Body {i + 1} ', style=label_style),
                html.Label('Position (x, y):', style=label_style),
                dcc.Input(id={'type': 'x', 'index': i}, type='number', value=0, style=input_style),
                dcc.Input(id={'type': 'y', 'index': i}, type='number', value=0, style=input_style),
                html.Label(' Velocity (vx, vy):', style=label_style),
                dcc.Input(id={'type': 'vx', 'index': i}, type='number', value=0, style=input_style),
                dcc.Input(id={'type': 'vy', 'index': i}, type='number', value=0, style=input_style),
                html.Label(' Mass:', style=label_style),
                dcc.Input(id={'type': 'mass', 'index': i}, type='number', value=1, style=input_style),
            ], style={"display": "flex", "align-items": "center", "margin-bottom": "10px"})
        )
    return inputs


######################################################################################################################

######################################################################################################################
@callback(
    Output('scatter-plot', 'figure'),
    Input({'type': 'x', 'index': dash.dependencies.ALL}, 'value'),
    Input({'type': 'y', 'index': dash.dependencies.ALL}, 'value'),
    Input({'type': 'vx', 'index': dash.dependencies.ALL}, 'value'),
    Input({'type': 'vy', 'index': dash.dependencies.ALL}, 'value')
)
def update_scatter_plot(x_values, y_values, vx_values, vy_values):
    fig = go.Figure(data=go.Scatter(x=x_values, y=y_values, mode='markers'))
    for x, y, vx, vy in zip(x_values, y_values, vx_values, vy_values):
        scale = 1
        vx *= scale
        vy *= scale
        fig.add_trace(
            go.Scatter(x=[x, x + vx], y=[y, y + vy], mode='lines')
        )

    fig.update_xaxes(range=[-5, 5])
    fig.update_yaxes(range=[-5, 5])
    fig.update_xaxes(scaleanchor='y', scaleratio=1)

    fig.update_layout(
      width=400,
      height=400, 
      margin=dict(l=20, r=20, b=20, t=20),
      showlegend=False,
      template="plotly_dark",
      xaxis=dict(showticklabels=True, zeroline=True, showline=True, showgrid=True),
      yaxis=dict(showticklabels=True, zeroline=True, showline=True, showgrid=True),
    )
    return fig
######################################################################################################################

######################################################################################################################
@callback(
    [
        Output('confirmation-message', 'children'),
        Output('bodies-container', 'children'),
        Output('bodies-input', 'children'),
        Output('express-plot-container', 'children')
     ],

    Input('submit-button', 'n_clicks'),

    State({'type': 'x', 'index': dash.dependencies.ALL}, 'value'),
    State({'type': 'y', 'index': dash.dependencies.ALL}, 'value'),
    State({'type': 'vx', 'index': dash.dependencies.ALL}, 'value'),
    State({'type': 'vy', 'index': dash.dependencies.ALL}, 'value'),
    State({'type': 'mass', 'index': dash.dependencies.ALL}, 'value'),
    
    prevent_initial_call=True
)
def submit_data(f_clicks, x_values, y_values, vx_values, vy_values, mass_values):    

    global bodies_data
    bodies_data = [
        {'x': x, 'y': y, 'vx': vx, 'vy': vy, 'mass': mass}
        for x, y, vx, vy, mass in zip(x_values, y_values, vx_values, vy_values, mass_values)
    ]
    data, mass = df(bodies_data)
    if [data,mass] == [
        [
            [-0.97000436, 0.97000436, 0.0],
            [0.24308753, -0.24308753, 0.0],
            [0.466203685, 0.466203685, -0.93240737],
            [0.43236573, 0.43236573, -0.86473146]
        ], 
      [1,1,1]
    ]:
        with open(r'C:\Users\User\REAL APP\data\infinity.json') as file:
            temp = json.load(file)
        fig = adjust(go.Figure(temp))
        return "Data submitted successfully!", None, None, dcc.Graph(figure=fig)
    
    elif [data,mass] == [
            [
                [-1, 1, 0],
                [0, 0, 0],
                [0.513938, 0.513938, -1.027876],
                [0.304736, 0.304736, -0.609472]
            ],
        [1,1,1]
    ]:
        with open(r'C:\Users\User\REAL APP\data\yinyang.json') as file:
            temp = json.load(file)
        fig = adjust(go.Figure(temp))
        return "Data submitted successfully!", None, None, dcc.Graph(figure=fig)
    
    elif [data,mass] == [
            [
                [-0.5, 0.5, 0.0207067154],
                [0, 0, 0.3133550361],
                [0, 0, 0],
                [0, 0, 0]
            ],
        [1,1,1]
    ]:
        with open(r'C:\Users\User\REAL APP\data\freefall.json') as file:
            temp = json.load(file)
        fig = adjust(go.Figure(temp))
        return "Data submitted successfully!", None, None, dcc.Graph(figure=fig)
    
    elif [data,mass] == [
            [
                [-1, 1, 0],  
                [0, 0, 0], 
                [0.2374365149, 0.2374365149, -0.9497460596], 
                [0.2536896353, 0.2536896353, -1.0147585412]
            ],
        [1,1,0.5]
    ]:
        with open(r'C:\Users\User\REAL APP\data\uneqmass.json') as file:
            temp = json.load(file)
        fig = adjust(go.Figure(temp))
        return "Data submitted successfully!", None, None, dcc.Graph(figure=fig)     
    else:
        fig = plot(data,mass)
        return "Data submitted successfully!", None, None, fig   
######################################################################################################################

######################################################################################################################
# @callback(
#     Output('express-plot-container', 'children'),
#     Input('next-button', 'n_clicks'),
#     prevent_initial_call=True
# )
def adjust(fig):
    fig.update_xaxes(
        scaleanchor='y',
        scaleratio=1
    )

    # You can update the layout here if needed
    fig.update_layout(
        width=1000,  # Set a specific width
        height=800,  # Set a specific height
        margin=dict(l=20, r=20, b=20, t=20),  # Adjust the margins
    )
    return fig
def df(bodies_data):
    initial_data = [[],[],[],[]]
    masses = []

    for d in bodies_data:
        initial_data[0].append(d['x'])
        initial_data[1].append(d['y'])
        initial_data[2].append(d['vx'])
        initial_data[3].append(d['vy'])
        masses.append(d['mass'])
    return initial_data, masses#

    

def plot(initial_data,masses):
    n = len(initial_data[0])

    x,y = compute(initial_data,masses)
    data = []

    time = list(range(len(x[0])))
    mass_data = []
    for m in masses:
        mass_data += [m] * len(x[0])

    for i in range(n):
        X,Y = x[i],y[i]
        obj = [f'body_{i+1}'] * len(x[0])
        df = pd.DataFrame(list(zip(obj,time,X,Y,mass_data)), columns=['body','time','x','y','masses'])
        data.append(df)
    df = pd.concat(data,axis=0)

    fig = px.scatter(df, x="x", y="y", animation_frame="time", animation_group="body",
            hover_name="body", color='body', range_x=[-5,5], range_y=[-5,5], template='plotly_dark',
            hover_data=['masses'])
    fig.update_traces(hovertemplate='x: %{x} <br>y: %{y}')
    fig.update_layout(
        xaxis=dict(showticklabels=False, zeroline=False, showline=False, showgrid=True),
        yaxis=dict(showticklabels=False, zeroline=False, showline=False, showgrid=True),
    )
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5
    fig.update_xaxes(
        scaleanchor='y',
        scaleratio=1
    )

    # You can update the layout here if needed
    fig.update_layout(
        width=1000,  # Set a specific width
        height=800,  # Set a specific height
        margin=dict(l=20, r=20, b=20, t=20),  # Adjust the margins
    )
    return dcc.Graph(figure=fig)  # Return the new Graph component
######################################################################################################################