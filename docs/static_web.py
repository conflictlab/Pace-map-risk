# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 22:24:43 2024

@author: thoma
"""
import dash
import pandas as pd
from dash import html, dcc
import dash_bootstrap_components as dbc

# Define the path for the new PDF and the uploaded image

title_text = 'Patterns of Conflict'

webapp = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP,dbc.themes.LUX],
                    meta_tags=[{'name': 'viewport',
                                'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}])
webapp.title = 'Pace Risk Map'
webapp._favicon = ("icone_pace.ico")
server = webapp.server
config = {'displayModeBar': False}

# App layout
webapp.layout = html.Div([
    html.Div([
        dbc.Container([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.H2(title_text, style={'textAlign': 'left','width':'80%'}),
                        dbc.Nav([
                            dbc.NavLink("Interactive Map",id='hidden_but',href="https://pace-risk-map-x35exdywcq-ue.a.run.app/", style={'color': '#555','display':'none'}),
                            dbc.NavLink("The Team", href="https://paceconflictlab.wixsite.com/conflict-research-la/team-4", style={'color': '#555'}),
                            dbc.NavLink("Contact", href="mailto:schincat@tcd.ie", style={'color': '#555'})
                        ]),
                    ]), lg=12, md=12, sm=12)
                ], style={'backgroundColor': '#D3D3D3', 'padding': '8px', 'marginBottom': '2vh'})
        ], fluid=True),
        html.Div([
            html.A(html.Img(src='Images/PaCE_final_icon.png', style={'height': '5vw', 'width': '5vw', 'marginLeft': '1vw'}), href='https://paceconflictlab.wixsite.com/conflict-research-la'),
            html.A(html.Img(src='Images/github-mark.png', style={'height': '5vw', 'width': '5vw', 'marginLeft': '1vw'}), href='https://github.com/ThomasSchinca/shapefinder_live'),
            html.A(html.Img(src='Images/x_logo.png', style={'height': '5vw', 'width': '5vw', 'marginLeft': '1vw'}), href='https://twitter.com/LabConflict')
        ], style={'position': 'absolute', 'right': '3vw', 'top': '1vh'}),
    dbc.Container(fluid=True, children=[
        dbc.Row([
            html.H3('Fatalities Risk Map',style= {'marginBottom': '5vh','marginTop':'2vh','textAlign': 'center'})
            ]),
        dbc.Row([
            dbc.Col(dcc.Markdown('''Our Global Risk Prediction Map identifies countries with similar past experiences in conflict-related
fatalities. By analyzing historical data patterns, this approach forecasts future trends and highlights
nations with comparable conflict trajectories.'''), 
                    width=12, style={'marginLeft': '5vw', 'width': '90vw'},id='parag')
        ]),
        
        dbc.Row([html.Div(html.Img(src='Images/map.png', style={'width': '80%'}), style={'text-align': 'center'})]),
        dbc.Row([
               html.H3('Global expected Fatalities',style={'marginBottom': '5vh','marginTop': '15vh','textAlign': 'center'})
               ]),     
        dbc.Row([html.Div(html.Img(src='Images/sub1_1.png', style={'width': '80%'}), style={'text-align': 'center'})]),
        dbc.Row([
               html.H3('Higher risk - Individual Cases',style={'marginBottom': '5vh','marginTop': '15vh','textAlign': 'center'})
               ]),  
        dbc.Row([
                dbc.Col([
                    html.Div(html.Img(src='Images/ex1.png', style={'height':'50vh'}), style={'text-align': 'center'})
                    ],style={'marginLeft': '10vw','width':'40vw'}),
                dbc.Col([
                    html.Div(html.Img(src='Images/ex1_sce.png', style={'height':'50vh'}), style={'text-align': 'center'})
                    ],style={'marginLeft': '5vw','width':'15vw'}),
                    dbc.Col([
                        html.Div(html.Img(src='Images/ex1_sce_p.png', style={'height':'50vh'}), style={'text-align': 'center'})
                        ],style={'marginLeft': '1vw','width':'15vw'})
                ],style={'marginBottom': '10vh'}),
        dbc.Row([
                dbc.Col([
                    html.Div(html.Img(src='Images/ex2.png', style={'height':'50vh'}), style={'text-align': 'center'})
                    ],style={'marginLeft': '10vw','width':'40vw'}),
                dbc.Col([
                    html.Div(html.Img(src='Images/ex2_sce.png', style={'height':'50vh'}), style={'text-align': 'center'})
                    ],style={'marginLeft': '5vw','width':'15vw'}),
                    dbc.Col([
                        html.Div(html.Img(src='Images/ex2_sce_p.png', style={'height':'50vh'}), style={'text-align': 'center'})
                        ],style={'marginLeft': '1vw','width':'15vw'})
                ],style={'marginBottom': '10vh'}),
        
        dbc.Row([
                dbc.Col([
                    html.Div(html.Img(src='Images/ex3.png', style={'height':'50vh'}), style={'text-align': 'center'})
                    ],style={'marginLeft': '10vw','width':'40vw'}),
                dbc.Col([
                    html.Div(html.Img(src='Images/ex3_sce.png', style={'height':'50vh'}), style={'text-align': 'center'})
                    ],style={'marginLeft': '5vw','width':'15vw'}),
                    dbc.Col([
                        html.Div(html.Img(src='Images/ex3_sce_p.png', style={'height':'50vh'}), style={'text-align': 'center'})
                        ],style={'marginLeft': '1vw','width':'15vw'})
                ],style={'marginBottom': '5vh'}),
        ]),  
    ]),
    html.Div([
        html.H1("About", style={'marginBottom':'5vh','marginTop':'10vh','textAlign': 'center'}),
        html.H3("Overview"),
        dcc.Markdown("""
        The "Patterns of Conflict" report identifies and compares conflict patterns across various
        countries. This process involves aggregating historical conflict data and matching similar
        patterns of conflict-related events. The methodology focuses on identifying trends and 
        potential future scenarios based on historical data. The objective is to provide a predictive 
        insight into how conflict patterns may evolve, aiding in better-informed strategic planning 
        and decision-making.

        The methodology in the "Patterns of Conflict" report is centered on a comparative analysis of 
        conflict-related data across countries. It involves the following steps:

        1.  Data collection. The data used in the "Patterns of Conflict" report is sourced from the 
            Uppsala Conflict Data Program (UCDP), a comprehensive database that records and codes 
            data on conflict and associated events worldwide. Specifically, the report makes use of 
            the "best" estimate variable for battle-related deaths provided by UCDP 
            (see https://ucdp.uu.se/downloads/brd/ucdp-brd-codebook.pdf)

        2.  Short sequences of casualty data are compared to each other using various algorithms 
            (DTW, Euclidean distance), which allow us to identify similar shapes in the data, even 
            ones that may be out of sync temporally. A distance threshold is applied to select only 
            sequences that are close matches.

        3.  The model then predicts potential increases or decreases in conflict-related fatalities 
            based on an average of past patterns.
        """),
        html.H3("Data Sources",style={'marginTop':80}),
        dcc.Markdown("""
        - **Conflict-Fatalities:**[UCDP Dataset](https://ucdp.uu.se/downloads/), aggregated at the country-monthly level.
        - **UCDP Georeferenced Event Dataset.**
        - **UCDP Candidate Events Dataset (to get the latest data).** """),
        html.H3("The Model",style={'marginTop':30}),
        dcc.Markdown("""
        The applied model operates by examining recent events within a country and aligning them with historical occurrences. It discerns patterns in the temporal evolution of incidents, enabling the identification of analogous scenarios from the past. This matching process contributes to a comprehensive understanding of when and where comparable situations have historically manifested. Consequently, the model plays a pivotal role in predicting the future trajectory of potential conflict-related scenarios based on these historical parallels, called ‘Past Future’."""),
        html.Div(html.Img(src='Images/explic.gif', style={'width': '80%'}), style={'text-align': 'center'}),
        html.H3("Find Historical Match",style={'marginTop':30}),
        dcc.Markdown("""
        To identify match in historical sequences, we employ dynamic time warping (DTW) distance. In contrast to the Euclidean distance, DTW offers greater flexibility in accommodating variations in time and window length. 
        DTW works by aligning the two sequences in a way that minimizes the total distance between corresponding points, allowing for both temporal shifts and local deformations. This alignment is achieved by warping the time axis of one sequence with respect to the other. The warping path represents the optimal alignment, and the DTW distance is the cumulative sum of the distances along this path.
        One of the key advantages of DTW is its ability to handle sequences of unequal length and to flexibly adapt to local variations in timing.
        The DTW distance is computed, and if it falls below a predefined threshold, the historical sequence is classified as a match."""),
        html.Div(html.Img(src='Images/dtw.gif', style={'width': '80%'}), style={'text-align': 'center'})
    ],style={'marginLeft':50})
])


if __name__ == '__main__':
    webapp.run_server(debug=False)#,host='0.0.0.0',port=8080)
    