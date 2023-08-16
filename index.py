# Import necessary libraries 
from dash import html, dcc
from dash.dependencies import Input, Output

# Connect to main app.py file
from app import app

# Connect to your app pages
from pages import page1, page2, page3, home

# Connect the navbar to the index
from components import navbar

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

# Run the app on localhost:8050
if __name__ == '__main__':
    app.run_server(debug=True)