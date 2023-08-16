# Import necessary libraries
from dash import html
import dash_bootstrap_components as dbc


# Define the navbar structure
def Navbar():

    layout = html.Div([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("[ Orbits ]", href="/orbits")),
                dbc.NavItem(dbc.NavLink("[ Ptolemy ]", href="/ptolemy")),
                dbc.NavItem(dbc.NavLink("[ The N-Body Problem ]", href="/nbody")),
            ] ,
            brand="BPhO Computational Challenge 2023",
            brand_href="/",
            color="dark",
            dark=True,
        ), 
    ])

    return layout