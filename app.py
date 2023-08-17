# Define the Dash App and it's properties here 

import dash
import dash_bootstrap_components as dbc
# Import necessary libraries 
from dash import html, dcc
from dash.dependencies import Input, Output

# Connect to your app pages
from pages import page1, page2, page3, home

# Connect the navbar to the index
from components import navbar


app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP], 
                meta_tags=[{"name": "viewport", "content": "width=device-width"}],
                suppress_callback_exceptions=True)

server = app.server

# define the navbar
nav = navbar.Navbar()

# Define the index page layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    nav, 
    html.Div(id='page-content', children=[]),
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/nbody':
        return page1.layout
    if pathname == '/orbits':
        return page2.layout
    if pathname == '/ptolemy':
        return page3.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True)
