import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import plotly.io as pio
import json

def compute(years,num_steps):
    #Constants:
    G = 6.67430e-11
    M = 1.989e30

    #Time array:
    t_start = 0.0
    t_end = years * 365 * 24 * 3600
    t = np.linspace(t_start, t_end, num_steps)
    dt = t[1] - t[0]

    #Initial states for the planets:
    initial_data = {
        "x": (5.79e10, 1.08e11, 1.5e11, 2.28e11, 7.78e11, 1.43e12, 2.87e12, 4.5e12, 4.44e12),
        "y": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "vx": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "vy": (4.79e4, 3.5e4, 2.98e4, 2.41e4, 1.31e4, 9.68e3, 6.8e3, 5.43e3, 6.13e3)
    }

    def acceleration(x, y):
        r = np.sqrt(x**2 + y**2)
        ax = -G * M * x / r**3  #Horizontal component of acceleration
        ay = -G * M * y / r**3  #Vertical component of acceleration
        return ax, ay

    def verlet_solver(x0,y0,vx0,vy0,num_steps):

        x = [0 for i in range(num_steps)]
        y = [0 for i in range(num_steps)]

        x[0], y[0] = x0, y0
        x[1], y[1] = x0 + vx0 * dt, y0 + vy0 * dt          #Verlet integration requires initial values at 2 conescutive points

        for i in range(1, num_steps - 1):

            ax, ay = acceleration(x[i], y[i])

            x[i + 1] = 2 * x[i] - x[i - 1] + ax * dt**2
            y[i + 1] = 2 * y[i] - y[i - 1] + ay * dt**2 #Sun is fixed at origin - positions are Cartesian coordinates. 
        
        return x, y

    n = len(initial_data['x'])
    data = [[] for i in range(n)]

    for i in range(n):                  # Runs 9x 
        x0 = initial_data["x"][i]   
        y0 = initial_data['y'][i]
        vx0 = initial_data['vx'][i] 
        vy0 = initial_data['vy'][i]     # Sets initial conditions 
        
        x,y = verlet_solver(x0,y0,vx0,vy0,num_steps) # Solves for Cartesian position
        data[i] = [x,y]
                                # Stores in 2D list - each sublists holds (x,y) for each time step
    xx = []
    yy = []

    for sub in data:
        xx += [sub[0]]
        yy += [sub[1]]  # Stores x and y in separate 2D lists
                        # Each sublist corresponds to a planet
            
    return xx,yy




#plot orbits
#############################################################################################################################################################################
def create_lines():
    n = 9
    s = []
    figs=[]

    for i in range(n):
        args = [[0.5,500],[1,500],[1.008,500],[3,500],[14,500],[30,500],[90,500],[200,500],[300,500]]
        years = args[i][0]
        steps = args[i][1]
        x,y = compute(years,steps)

        planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
        t_start = 0.0
        t_end = years * 365 * 24 * 3600
        t = np.linspace(t_start, t_end, steps)
        dt = t[1] - t[0]
        
        colors = ["red", "orange", "blue", "red", "orange", "yellow", "blue", "darkblue", "pink"]
        # colors = [[color] * steps for color in colors]

        X,Y = x[i],y[i]
        obj = [f'body_{i+1}'] * len(x[0])
        zeros = [0,] * steps
        df = pd.DataFrame(list(zip(obj,t,X,Y,zeros)), columns=['body','time','x','y','zeros'])
        s.append(df)
        df = pd.concat(s,axis=0)

        fig = px.line(df[i*steps:(i+1)*steps], x='x', y='y',template='plotly_dark')
        fig.update_traces(line=dict(color=colors[i]))

        fig.update_traces(hovertemplate=planets[i]+': <br>x: %{x} <br>y: %{y}')

        figs.append(fig)
    return figs

#############################################################################################################################################################################
main = px.scatter(template='plotly_dark')
figs = create_lines()
for f in figs:
    main.add_trace(f.data[0])
main.update_layout(
    xaxis=dict(showticklabels=True, zeroline=False, showline=False, showgrid=True),
    yaxis=dict(showticklabels=True, zeroline=False, showline=False, showgrid=True),
)
main.update_xaxes(
    scaleanchor='y',
    scaleratio=1
)
main.update_layout(
    width=1880,  # Set a specific width
    height=900,  # Set a specific height
    margin=dict(l=20, r=20, b=20, t=20),)  # Adjust the margins

####################################################################################

layout = html.Div(
        [
    dcc.Loading(
        id='loading', type='default', color='aliceblue', children=
                [
                html.Div(id='express-plot-container', children=[
                            dcc.Graph(figure=main),
                        ]
                    )
                ]),

        ],
    )
