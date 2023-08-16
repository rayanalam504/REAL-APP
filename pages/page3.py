# import dash
# from dash import html, dcc, Input, Output, State, callback
# import plotly.graph_objects as go
# import plotly.express as px
# import numpy as np
# from scipy.integrate import solve_ivp
# import pandas as pd
# import plotly.io as pio
# import json

# def compute(years,num_steps):
#     #Constants:
#     G = 6.67430e-11
#     M = 1.989e30

#     #Time array:
#     t_start = 0.0
#     t_end = years * 365 * 24 * 3600
#     t = np.linspace(t_start, t_end, num_steps)
#     dt = t[1] - t[0]

#     #Initial states for the planets:
#     initial_data = {
#         "x": (5.79e10, 1.08e11, 1.5e11, 2.28e11, 7.78e11, 1.43e12, 2.87e12, 4.5e12, 4.44e12),
#         "y": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
#         "vx": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
#         "vy": (4.79e4, 3.5e4, 2.98e4, 2.41e4, 1.31e4, 9.68e3, 6.8e3, 5.43e3, 6.13e3)
#     }

#     def acceleration(x, y):
#         r = np.sqrt(x**2 + y**2)
#         ax = -G * M * x / r**3  #Horizontal component of acceleration
#         ay = -G * M * y / r**3  #Vertical component of acceleration
#         return ax, ay

#     def verlet_solver(x0,y0,vx0,vy0,num_steps):

#         x = [0 for i in range(num_steps)]
#         y = [0 for i in range(num_steps)]

#         x[0], y[0] = x0, y0
#         x[1], y[1] = x0 + vx0 * dt, y0 + vy0 * dt          #Verlet integration requires initial values at 2 conescutive points

#         for i in range(1, num_steps - 1):

#             ax, ay = acceleration(x[i], y[i])

#             x[i + 1] = 2 * x[i] - x[i - 1] + ax * dt**2
#             y[i + 1] = 2 * y[i] - y[i - 1] + ay * dt**2 #Sun is fixed at origin - positions are Cartesian coordinates. 
        
#         return x, y

#     n = len(initial_data['x'])
#     data = [[] for i in range(n)]

#     for i in range(n):                  # Runs 9x 
#         x0 = initial_data["x"][i]   
#         y0 = initial_data['y'][i]
#         vx0 = initial_data['vx'][i] 
#         vy0 = initial_data['vy'][i]     # Sets initial conditions 
        
#         x,y = verlet_solver(x0,y0,vx0,vy0,num_steps) # Solves for Cartesian position
#         data[i] = [x,y]
#                                 # Stores in 2D list - each sublists holds (x,y) for each time step
#     xx = []
#     yy = []

#     for sub in data:
#         xx += [sub[0]]
#         yy += [sub[1]]  # Stores x and y in separate 2D lists
#                         # Each sublist corresponds to a planet
#     return xx,yy

# #plot orbits
# #############################################################################################################################################################################
# def create_lines():
#     n = 9
#     s = []
#     figs=[]

#     for i in range(n):
#         args = [[0.5,500],[1,500],[1.008,500],[3,500],[14,500],[30,500],[90,500],[200,500],[300,500]]
#         years = args[i][0]
#         steps = args[i][1]
#         x,y = compute(years,steps)

#         planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
#         t_start = 0.0
#         t_end = years * 365 * 24 * 3600
#         t = np.linspace(t_start, t_end, steps)
#         dt = t[1] - t[0]
        
#         colors = ["red", "orange", "blue", "red", "orange", "yellow", "blue", "darkblue", "pink"]
#         # colors = [[color] * steps for color in colors]

#         X,Y = x[i],y[i]
#         obj = [f'body_{i+1}'] * len(x[0])
#         zeros = [0,] * steps
#         df = pd.DataFrame(list(zip(obj,t,X,Y,zeros)), columns=['body','time','x','y','zeros'])
#         s.append(df)
#         df = pd.concat(s,axis=0)

#         fig = px.line(df[i*steps:(i+1)*steps], x='x', y='y',template='plotly_dark')
#         fig.update_traces(line=dict(color=colors[i]))

#         fig.update_traces(hovertemplate=planets[i]+': <br>x: %{x} <br>y: %{y}')

#         figs.append(fig)
#     return figs

# #############################################################################################################################################################################
# main = px.scatter(template='plotly_dark')
# main.update_layout(
#     autosize=True,
#     # margin=dict(t=0, b=0, l=0, r=0),
#     # autosize=True,
#     width=1880,  # specify the width you want
#     height=900,  # specify the height you want
#     margin=dict(t=50, b=50, l=50, r=50)
# )
# main.update_layout(
#     xaxis=dict(range=[-5e12, 5e12]),  # Replace xmin and xmax with your desired values
#     yaxis=dict(range=[-5e12, 5e12])   # Replace ymin and ymax with your desired values
# )
# # main.update_xaxes(
# #     scaleanchor='y',
# #     scaleratio=1
# # )
# ####################################################################################

# layout = html.Div(
#         [
#     dcc.Loading(
#         id='loading', type='default', color='aliceblue', children=
#                 [
#                 html.Div(id='graph-container', children=[
#                             dcc.Graph(figure=main),
#                         ], style={'height': '100vh', 'width': '100vw'}
#                     )
#                 ]),
#           html.Div([
#             html.Button('Mercury', id='mercury', style={'width': '120px'}),
#             html.Button('Venus', id='venus', style={'width': '120px'}),
#             html.Button('Earth', id='earth', style={'width': '120px'}),
#             html.Button('Mars', id='mars', style={'width': '120px'}),
#             html.Button('Jupiter', id='jupiter', style={'width': '120px'}),
#             html.Button('Saturn', id='saturn', style={'width': '120px'}),
#             html.Button('Uranus', id='uranus', style={'width': '120px'}),
#             html.Button('Neptune', id='neptune', style={'width': '120px'}),
#             html.Button('Pluto', id='pluto', style={'width': '120px'})
#             ], 
#             id='button-group', style={
#             'position': 'absolute',  # This will overlay the button group on top of the plot
#             'top': '100px',
#             'left': '70px',
#             'z-index': 1000,  # Ensure the button group appears above the plot
#             'display': 'flex',
#             'flexDirection': 'column',
#             'gap': '10px',
#             'background-color': '#222222',  # Semi-transparent background for visibility
#             'padding': '10px',
#             'border-radius': '5px',
#             'border-color': '#000000'
#           }
#         )
#       ],
#     )

# # @callback(
# #     Output('graph-container', 'children'),
# #     [Input('mercury', 'n_clicks'),
# #      Input('venus', 'n_clicks'),
# #      Input('earth', 'n_clicks'),
# #      Input('mars', 'n_clicks'),
# #      Input('jupiter', 'n_clicks'),
# #      Input('saturn', 'n_clicks'),
# #      Input('uranus', 'n_clicks'),
# #      Input('neptune', 'n_clicks'),
# #      Input('pluto', 'n_clicks')]
# # )

# # def create_body_inputs(m,v,e,ma,j,s,u,n,p):
# #   s = []
# #   figs = []
# #   n=9
# #   for i in range(n):
# #     args = [[0.5,500],[1,500],[1.008,500],[3,500],[14,500],[30,500],[90,500],[200,500],[300,500]]
# #     years = args[i][0]
# #     steps = args[i][1]
# #     x,y = compute(years,steps)

# #     planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
# #     t_start = 0.0
# #     t_end = years * 365 * 24 * 3600
# #     t = np.linspace(t_start, t_end, steps)
# #     dt = t[1] - t[0]
    
# #     colors = ["red", "orange", "blue", "red", "orange", "yellow", "blue", "darkblue", "pink"]
# #     # colors = [[color] * steps for color in colors]

# #     X,Y = x[i],y[i]
# #     obj = [f'body_{i+1}'] * len(x[0])
# #     zeros = [0,] * steps
# #     df = pd.DataFrame(list(zip(obj,t,X,Y,zeros)), columns=['body','time','x','y','zeros'])
# #     s.append(df)
# #   df = pd.concat(s,axis=0)

# #   fig = px.line(df[i*steps:(i+1)*steps], x='x', y='y',template='plotly_dark')
# #   fig.update_traces(line=dict(color=colors[i]))

# #   fig.update_traces(hovertemplate=planets[i]+': <br>x: %{x} <br>y: %{y}')

# #   figs.append(fig)
  

# #   for f in figs:
# #       main.add_trace(f.data[0])

# #   main.update_layout(
# #       xaxis=dict(showticklabels=True, zeroline=False, showline=False, showgrid=True),
# #       yaxis=dict(showticklabels=True, zeroline=False, showline=False, showgrid=True),
# #   )
# #   main.update_xaxes(
# #       scaleanchor='y',
# #       scaleratio=1
# #   )
# #   main.update_layout(
# #       width=1880,  # Set a specific width
# #       height=900,  # Set a specific height
# #       margin=dict(l=20, r=20, b=20, t=20),)
      
# #   return main
# def ptol():
#     n = 9
#     s = []
#     figs=[]
#     a = 2
#     for i in range(n):
#         args = [[0.5,500],[1,500],[1.008,500],[3,500],[14,500],[30,500],[90,500],[200,500],[300,500]]
#         years = args[i][0]
#         steps = args[i][1]
#         x,y = compute(years,steps)

#         planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
#         t_start = 0.0
#         t_end = years * 365 * 24 * 3600
#         t = np.linspace(t_start, t_end, steps)
#         dt = t[1] - t[0]
        
#         colors = ["red", "orange", "blue", "red", "orange", "yellow", "blue", "darkblue", "pink"]
#         # colors = [[color] * steps for color in colors]
#         Xa,Ya = x[a],y[a]
#         X,Y = x[i],y[i]
#         obj = [f'body_{i+1}'] * len(x[0])
#         zeros = [0,] * steps
#         df = pd.DataFrame(list(zip(obj,t,X,Y,zeros)), columns=['body','time','x','y','zeros'])
#         s.append(df)
#         df = pd.concat(s,axis=0)

#         fig = px.line(df[i*steps:(i+1)*steps], x='x', y='y',template='plotly_dark')
#         fig.update_traces(line=dict(color=colors[i]))

#         fig.update_traces(hovertemplate=planets[i]+': <br>x: %{x} <br>y: %{y}')

#         figs.append(fig)
#     return figs
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
# def create_lines():
#     n = 9
#     s = []
#     figs=[]
#     a=3
#     for i in range(n):
#         args = [[10,5000],[10,5000],[15,5000],[30,5000],[70,5000],[90,5000],[150,5000],[200,5000],[500,10000]]
#         years = args[i][0]
#         steps = args[i][1]
#         x,y = compute(years,steps)

#         planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
#         t_start = 0.0
#         t_end = years * 365 * 24 * 3600
#         t = np.linspace(t_start, t_end, steps)
#         dt = t[1] - t[0]
        
#         colors = ["red", "orange", "blue", "red", "orange", "yellow", "blue", "darkblue", "pink"]
#         # colors = [[color] * steps for color in colors]

#         X,Y = x[i],y[i]
#         Xa,Ya = x[a],y[a]
#         for j in range(len(X)):
#             X[j] -= Xa[j]
#             Y[j] -= Ya[j]
            
#         obj = [f'body_{i+1}'] * len(x[0])
#         zeros = [0,] * steps
#         df = pd.DataFrame(list(zip(obj,t,X,Y,zeros)), columns=['body','time','x','y','zeros'])
#         s.append(df)
#         df = pd.concat(s,axis=0)

#         fig = px.line(df[i*steps:(i+1)*steps], x='x', y='y',template='plotly_dark')
#         fig.update_traces(line=dict(color=colors[i]))

#         fig.update_traces(hovertemplate=planets[i]+': <br>x: %{x} <br>y: %{y}')

#         figs.append(fig)
#     return figs

#############################################################################################################################################################################
# main = px.scatter(template='plotly_dark')
# figs = create_lines()
# for f in figs:
#     main.add_trace(f.data[0])
# main.update_layout(
#     xaxis=dict(showticklabels=True, zeroline=False, showline=False, showgrid=True),
#     yaxis=dict(showticklabels=True, zeroline=False, showline=False, showgrid=True),
# )
# main.update_xaxes(
#     scaleanchor='y',
#     scaleratio=1
# )
# main.update_layout(
#     width=1880,  # Set a specific width
#     height=900,  # Set a specific height
#     margin=dict(l=20, r=20, b=20, t=20),)  # Adjust the margins

####################################################################################


main = px.scatter(template='plotly_dark')

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
    margin=dict(l=20, r=20, b=20, t=20),)

common_button_style = {
    'width': '150px',  # Adjust width as required
}

layout = html.Div([
    dcc.Interval(
        id='onload-interval',
        interval=1*1000,  # in milliseconds
        max_intervals=1   # Intervals are fired once
    ),
    dcc.Loading(
        id='loading', 
        type='default', 
        color='aliceblue', 
        children=[
            html.Div(id='express', children=[
                dcc.Graph(id='id', figure=main),
            ]),
            html.Div(children=[
                html.Div([
                    html.Label('Pick a planet to '),
                    html.Br(),
                    html.Label('lock to the origin'),
                    html.Br(),
                    html.Button('Mercury', id='mercury', style=common_button_style)
                ], style={'padding-bottom': '10px'}),

                html.Div([
                    html.Button('Venus', id='venus', style=common_button_style)
                ], style={'padding-bottom': '10px'}),

                html.Div([
                    html.Button('Earth', id='earth', style=common_button_style)
                ], style={'padding-bottom': '10px'}),

                html.Div([
                    html.Button('Mars', id='mars', style=common_button_style)
                ], style={'padding-bottom': '10px'}),

                html.Div([
                    html.Button('Jupiter', id='jupiter', style=common_button_style)
                ], style={'padding-bottom': '10px'}),

                html.Div([
                    html.Button('Saturn', id='saturn', style=common_button_style)
                ], style={'padding-bottom': '10px'}),

                html.Div([
                    html.Button('Uranus', id='uranus', style=common_button_style)
                ], style={'padding-bottom': '10px'}),

                html.Div([
                    html.Button('Neptune', id='neptune', style=common_button_style)
                ], style={'padding-bottom': '10px'}),
                html.Label('Drag the cursor'),
                html.Br(),
                html.Label('to zoom '),
            ], 
            style={
                'position': 'absolute',
                'top': '100px',
                'left': '70px',
                'background-color': '#111111',
                'box-shadow': '5px 5px 5px black'
            }),
        ]
    )
])

planet_dict = {
    "mercury": 0,
    "venus": 1,
    "earth": 2,
    "mars": 3,
    "jupiter": 4,
    "saturn": 5,
    "uranus": 6,
    "neptune": 7,
    "pluto": 8
}


@callback(
    Output('id', 'figure'),
    [
        Input('mercury', 'n_clicks'),
        Input('venus', 'n_clicks'),
        Input('earth',  'n_clicks'),
        Input('mars', 'n_clicks'),
        Input('jupiter', 'n_clicks'),
        Input('saturn',  'n_clicks'),   
        Input('uranus', 'n_clicks'),
        Input('neptune',  'n_clicks'),   
    ],
    prevent_initial_call = True
)
def func(a,b,c,d,e,f,g,h):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "No button has been pressed yet."
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    n = 9
    s = []
    figs=[]
    a=planet_dict[button_id]
    for i in range(n):
        args = [[20,10000],[20,10000],[30,10000],[60,10000],[140,10000],[180,10000],[300,10000],[400,10000],[500,100000]]
        years = args[i][0]
        steps = args[i][1]
        x,y = compute(years,steps)

        planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
        t_start = 0.0
        t_end = years * 365 * 24 * 3600
        t = np.linspace(t_start, t_end, steps)
        dt = t[1] - t[0]
        
        colors = ["pink", "orange", "blue", "red", "orange", "yellow", "blue", "darkblue", "pink"]
        
        X,Y = x[i],y[i]
        Xa,Ya = x[a],y[a]
        for j in range(len(X)):
            X[j] -= Xa[j]
            Y[j] -= Ya[j]
            
        obj = [f'body_{i+1}'] * len(x[0])
        zeros = [0,] * steps
        df = pd.DataFrame(list(zip(obj,t,X,Y,zeros)), columns=['body','time','x','y','zeros'])
        s.append(df)
        df = pd.concat(s,axis=0)

        fig = px.line(df[i*steps:(i+1)*steps], x='x', y='y',template='plotly_dark')
        fig.update_traces(line=dict(color=colors[i]))

        fig.update_traces(hovertemplate=planets[i]+': <br>x: %{x} <br>y: %{y}')

        figs.append(fig)
        
    main = px.scatter(template='plotly_dark')

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
        margin=dict(l=20, r=20, b=20, t=20),)
    
    return main