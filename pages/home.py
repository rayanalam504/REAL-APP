from dash import html

layout = html.Div(
    [
    html.Br(),
    html.Label('On this website we have pages for an interactive visualization of orbital mechanics and a customizable n-body simulator'),
    html.Br(),
    html.Label('This gives viewers the chance to examine our simulations in more detail and showcases our extension into n-body dynamics. '),
    html.Br(),
    html.Label('You can view preset stable configurations (or input your own...) and create chaotic ones.'),
    html.Br(),
    html.Label('Drag the cursor to zoom, double click to zoom out.')
    ]
)
