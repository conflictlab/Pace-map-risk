# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 22:24:43 2024

@author: thoma
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash.dependencies import Input, Output
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import json
import pandas as pd
import pickle 
import base64

def generate_subplot(series):
    fig = px.line(series, title=series.name)
    fig.update_layout(
                showlegend=False,plot_bgcolor="white",
                margin=dict(t=30,l=30,b=5,r=5),
                xaxis_title='',
                yaxis_title='')
    fig.update_traces(line=dict(color='grey'))
    return fig

def generate_subplot_sce(series,col):
    fig = px.line(series, title=series.name)
    fig.update_layout(
        showlegend=False, plot_bgcolor="white",
        margin=dict(t=30, l=30, b=5, r=5),
        xaxis_title='',
        yaxis_title='',
        title=dict(text=series.name, font=dict(color=col))
    )
    fig.update_traces(line=dict(color='grey'))
    last_six_points = series.tail(6)
    last_sept_points = series.tail(7)
    fig.add_trace(go.Scatter(
        x=last_six_points.index,
        y=last_six_points.values,
        mode='markers+lines',
        marker=dict(color=col),
        line=dict(color=col),
        name='Past Future'
    ))
    fig.add_trace(go.Scatter(
        x=last_sept_points.index,
        y=last_sept_points.values,
        mode='lines',
        marker=dict(color=col),
        line=dict(color=col),
        name='Past Future'
    ))
    return fig

def get_color(log_per_pr):
    cmap = plt.get_cmap('Reds')
    norm_log_per_pr = (log_per_pr - world['log_per_pred'].min()) / (world['log_per_pred'].max() - world['log_per_pred'].min())
    rgba_color = cmap(norm_log_per_pr)
    hex_color = to_hex(rgba_color)
    return hex_color


world = gpd.read_file('world_plot.geojson')
pred_df=pd.read_csv('Pred_df.csv',parse_dates=True,index_col=(0))
missing_columns = set(world['name']) - set(pred_df.columns)
pred_df = pred_df.assign(**{col: [0] * 6 for col in missing_columns})
pred_df.index = [f"t+{i}" for i in range(1,len(pred_df)+1)]
pred_df_min =pd.read_csv('Pred_df_min.csv',parse_dates=True,index_col=(0))
pred_df_min = pred_df_min.assign(**{col: [0] * 6 for col in missing_columns})
pred_df_max =pd.read_csv('Pred_df_max.csv',parse_dates=True,index_col=(0))
pred_df_max = pred_df_max.assign(**{col: [0] * 6 for col in missing_columns})
hist_df=pd.read_csv('Hist.csv',parse_dates=True,index_col=(0))
hist_df = hist_df.assign(**{col: [0] * 10 for col in missing_columns})
with open('saved_dictionary.pkl', 'rb') as f:
    dict_m_o = pickle.load(f)
rena={'Bosnia-Herzegovina':'Bosnia and Herz.','Cambodia (Kampuchea)':'Cambodia',
                                   'Central African Republic':'Central African Rep.','DR Congo (Zaire)':'Dem. Rep. Congo',
                                   'Ivory Coast':'Côte d\'Ivoire','Kingdom of eSwatini (Swaziland)':'eSwatini', 'Dominican Republic':'Dominican Rep.',
                                   'Macedonia, FYR':'Macedonia','Madagascar (Malagasy)':'Madagascar','Myanmar (Burma)':'Myanmar',
                                   'Russia (Soviet Union)':'Russia','Serbia (Yugoslavia)':'Serbia','South Sudan':'S. Sudan',
                                   'Yemen (North Yemen)':'Yemen','Zimbabwe (Rhodesia)':'Zimbabwe','Vietnam (North Vietnam)':'Vietnam'}
reversed_rena = {value: key for key, value in rena.items()}

dict_m = {rena[key] if key in rena else key: item for key, item in dict_m_o.items()}
dict_m.update({col: [] for col in missing_columns})

new_dict = {}
for key, series_int_list in dict_m.items():
    series_list = [item[0] for item in series_int_list[:15]]
    new_dict[key] = series_list
    
dist_dict = {}
for key, series_int_list in dict_m.items():
    series_list = [item[1] for item in series_int_list[:5]]
    dist_dict[key] = series_list

world['name_alt']= world['name'].replace(reversed_rena)
world['color'] = world['log_per_pred'].apply(get_color)
l_country = [dl.GeoJSON(data=json.loads(world.iloc[index:index+1].to_json()), style={'color': row['color'], 'opacity': 0, 'fillOpacity': '1','z-index':1}) for index, row in world.iterrows()]

df_perc = pd.read_csv('perc.csv',parse_dates=True,index_col=(0))
df_d = pd.read_csv('dec.csv',parse_dates=True,index_col=(0))
df_s = pd.read_csv('sta.csv',parse_dates=True,index_col=(0))
df_i = pd.read_csv('inc.csv',parse_dates=True,index_col=(0))

with open('dict_sce.pkl', 'rb') as f:
    dict_sce = pickle.load(f)
missing_columns.discard('Dominican Rep.')    
dict_sce.update({col: [[],[],[]] for col in missing_columns})


pace_png = base64.b64encode(open('PaCE_final_icon.png', 'rb').read()).decode('ascii')
git_png = base64.b64encode(open('github-mark.png', 'rb').read()).decode('ascii')
x_logo = base64.b64encode(open('x_logo.png', 'rb').read()).decode('ascii')
gif_fo = base64.b64encode(open('Images/explic.gif', 'rb').read()).decode('ascii')
gif_dtw = base64.b64encode(open('Images/dtw.gif', 'rb').read()).decode('ascii')
ab1 = base64.b64encode(open('Images/about_1.png', 'rb').read()).decode('ascii')
ab2 = base64.b64encode(open('Images/about_2.png', 'rb').read()).decode('ascii')
ab3 = base64.b64encode(open('Images/about_3.png', 'rb').read()).decode('ascii')
ab4 = base64.b64encode(open('Images/about_4.png', 'rb').read()).decode('ascii')


webapp = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP,dbc.themes.LUX],
                    meta_tags=[{'name': 'viewport',
                                'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}])
webapp.title = 'Pace Risk Map'
webapp._favicon = ("icone_pace.ico")
server = webapp.server
config = {'displayModeBar': False}

# App layout
home_layout = html.Div([
    dcc.Loading(
        id="loading-2",
    children=[
    html.Div([
        dbc.Container([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.H2("Fatalities Risk Map", style={'textAlign': 'left','width':'80%'}),
                        dbc.Nav([
                            dbc.NavLink("Try The Model", href="https://shapefinder.azurewebsites.net/", style={'color': '#555'}),
                            dbc.NavLink("Monthly Report", href="/report", style={'color': '#555'}),
                            dbc.NavLink("About", href="/about", style={'color': '#555'}),
                            dbc.NavLink("The Team", href="https://paceconflictlab.wixsite.com/conflict-research-la/team-4", style={'color': '#555'}),
                            dbc.NavLink("Contact", href="mailto:schincat@tcd.ie", style={'color': '#555'})
                        ]),
                    ]), lg=12, md=12, sm=12)
                ], style={'backgroundColor': '#D3D3D3', 'padding': '8px', 'marginBottom': '2vh'})
        ], fluid=True),
        html.Div([
            html.A(html.Img(src='data:image/png;base64,{}'.format(pace_png), style={'height': '5vw', 'width': '5vw', 'marginLeft': '1vw'}), href='https://paceconflictlab.wixsite.com/conflict-research-la'),
            html.A(html.Img(src='data:image/png;base64,{}'.format(git_png), style={'height': '5vw', 'width': '5vw', 'marginLeft': '1vw'}), href='https://github.com/ThomasSchinca/shapefinder_live'),
            html.A(html.Img(src='data:image/png;base64,{}'.format(x_logo), style={'height': '5vw', 'width': '5vw', 'marginLeft': '1vw'}), href='https://twitter.com/LabConflict')
        ], style={'position': 'absolute', 'right': '3vw', 'top': '1vh'}),
    dbc.Container(fluid=True, children=[
        dbc.Row([
            dbc.Col(dcc.Markdown('''The Fatalities Risk Map assesses conflict risks by analyzing historical conflict data. It predicts future trends in fatalities by identifying countries with similar conflict histories, thereby highlighting comparable risk trajectories. Click on a country to show country-specific data.'''), 
                    width=12, style={'marginLeft': '5vw', 'width': '90vw'},id='parag')
        ]),
        dbc.Row([
            dbc.Col(dl.Map(center=[0, 5], zoom=2, minZoom=2, children=l_country + [
                dl.GeoJSON(url='/assets/world_plot.geojson', id='total_c', style={'color': 'black', 'weight': 1, 'opacity': 1, 'fillOpacity': 0})
            ], style={'width': '95vw', 'height': '78vh', 'backgroundColor': '#F5F5F5','z-index':1}, zoomControl=False, attributionControl=False,maxBounds=[[-60, -220], [90, 350]], id='map'), 
                    width=12)
        ]),
        html.Div([
            html.Div(id='plot_test'),
            html.Div(id='plot_test2'),
            html.Div(id='plot_test3')
            ], id='responsive-div')
        ]),  
    ],style={'backgroundColor': '#F5F5F5'}),
    html.Div(id='space',style={'width':'100%','height': '5vh','backgroundColor': '#F5F5F5'},children=[
        dbc.Row([
            dbc.Col(dcc.Markdown('source data : [UCDP](https://ucdp.uu.se/downloads/)', style={
                'marginLeft':'40vw'}), width=12)
        ]),]),
    html.Div(id='plot_test4')],
    type="dot",fullscreen=True,color="#df2226"
    )
])

report_layout= html.Div([
    dbc.Container([
            dbc.Row([
                dbc.Col(html.Div([
                    html.H2("Report", style={'textAlign': 'left'}),
                    dbc.Nav([
                        dbc.NavLink("Risk Map", href="/home", style={'color': '#555'}),
                        dbc.NavLink("About", href="/about", style={'color': '#555'}),
                        dbc.NavLink("The Team", href="https://paceconflictlab.wixsite.com/conflict-research-la/team-4", style={'color': '#555'}),
                        dbc.NavLink("Contact", href="mailto:schincat@tcd.ie", style={'color': '#555'})
                    ]),
                    html.Div([
                        html.A(html.Img(src='data:image/png;base64,{}'.format(pace_png), style={'height': '5vw', 'width': '5vw', 'marginLeft': '1vw'}), href='https://paceconflictlab.wixsite.com/conflict-research-la'),
                        html.A(html.Img(src='data:image/png;base64,{}'.format(git_png), style={'height': '5vw', 'width': '5vw', 'marginLeft': '1vw'}), href='https://github.com/ThomasSchinca/shapefinder_live'),
                        html.A(html.Img(src='data:image/png;base64,{}'.format(x_logo), style={'height': '5vw', 'width': '5vw', 'marginLeft': '1vw'}), href='https://twitter.com/LabConflict')
                    ], style={'position': 'absolute', 'right': '3vw', 'top': '1vh'})
                ]), lg=12, md=12, sm=12)
            ], style={'backgroundColor': '#D3D3D3', 'padding': '8px', 'marginBottom': 20})
    ], fluid=True),
    html.Iframe(src='assets/Report.pdf', width='80%', height= '1000vh',style={'marginLeft':'10%'}),
])

about_layout=html.Div([
    dbc.Container([
            dbc.Row([
                dbc.Col(html.Div([
                    html.H2("About", style={'textAlign': 'left'}),
                    dbc.Nav([
                        dbc.NavLink("Risk Map", href="/home", style={'color': '#555'}),
                        dbc.NavLink("Monthly Report", href="/report", style={'color': '#555'}),
                        dbc.NavLink("The Team", href="https://paceconflictlab.wixsite.com/conflict-research-la/team-4", style={'color': '#555'}),
                        dbc.NavLink("Contact", href="mailto:schincat@tcd.ie", style={'color': '#555'})
                    ]),
                    html.Div([
                        html.A(html.Img(src='data:image/png;base64,{}'.format(pace_png), style={'height': '5vw', 'width': '5vw', 'marginLeft': '1vw'}), href='https://paceconflictlab.wixsite.com/conflict-research-la'),
                        html.A(html.Img(src='data:image/png;base64,{}'.format(git_png), style={'height': '5vw', 'width': '5vw', 'marginLeft': '1vw'}), href='https://github.com/ThomasSchinca/shapefinder_live'),
                        html.A(html.Img(src='data:image/png;base64,{}'.format(x_logo), style={'height': '5vw', 'width': '5vw', 'marginLeft': '1vw'}), href='https://twitter.com/LabConflict')
                    ], style={'position': 'absolute', 'right': '3vw', 'top': '1vh'})
                ]), lg=12, md=12, sm=12)
            ], style={'backgroundColor': '#D3D3D3', 'padding': '8px', 'marginBottom': 20})
    ], fluid=True),
    
    html.Div([
        html.H1("Pace Risk Map Web Application", style={'marginBottom':20,'textAlign': 'center'}),
        html.H3("Overview"),
        dcc.Markdown("""
        The Pace Risk Map web application is designed to visualize and analyze risk factors related to conflict and fatalities across different countries. It incorporates geographical data, matching models, and historical information to provide insights into potential conflict scenarios.
        """),
        html.H3("Purpose",style={'marginTop':30}),
        dcc.Markdown("""
        - Provide a user-friendly interface for exploring conflict risk data.
        - Display historical information for individual countries.
        - Offer additional resources for further exploration.
        
        """),
        html.H3("Functionality",style={'marginTop':30}),
        html.Div(html.Img(src='data:image/png;base64,{}'.format(ab1), style={'width': '80%'}), style={'text-align': 'center'}),
        html.Div(html.Img(src='data:image/png;base64,{}'.format(ab2), style={'width': '80%'}), style={'text-align': 'center'}),
        html.Div(html.Img(src='data:image/png;base64,{}'.format(ab3), style={'width': '80%','marginTop':80}), style={'text-align': 'center'}),
        html.Div(html.Img(src='data:image/png;base64,{}'.format(ab4), style={'width': '80%','marginTop':80}), style={'text-align': 'center'}),
        html.H3("Data Sources",style={'marginTop':80}),
        
        dcc.Markdown("""
        - **Conflict-Fatalities:**[UCDP Dataset](https://ucdp.uu.se/downloads/), aggregated at the country-monthly level.
        - **UCDP Georeferenced Event Dataset.**
        - **UCDP Candidate Events Dataset (to get the latest data).** """),
        html.H3("The Model",style={'marginTop':30}),
        dcc.Markdown("""
        The applied model operates by examining recent events within a country and aligning them with historical occurrences. It discerns patterns in the temporal evolution of incidents, enabling the identification of analogous scenarios from the past. This matching process contributes to a comprehensive understanding of when and where comparable situations have historically manifested. Consequently, the model plays a pivotal role in predicting the future trajectory of potential conflict-related scenarios based on these historical parallels, called ‘Past Future’."""),
        html.Div(html.Img(src='data:image/gif;base64,{}'.format(gif_fo), style={'width': '80%'}), style={'text-align': 'center'}),
        html.H3("Find Historical Match",style={'marginTop':30}),
        dcc.Markdown("""
        To identify match in historical sequences, we employ dynamic time warping (DTW) distance. In contrast to the Euclidean distance, DTW offers greater flexibility in accommodating variations in time and window length. 
        DTW works by aligning the two sequences in a way that minimizes the total distance between corresponding points, allowing for both temporal shifts and local deformations. This alignment is achieved by warping the time axis of one sequence with respect to the other. The warping path represents the optimal alignment, and the DTW distance is the cumulative sum of the distances along this path.
        One of the key advantages of DTW is its ability to handle sequences of unequal length and to flexibly adapt to local variations in timing.
        The DTW distance is computed, and if it falls below a predefined threshold, the historical sequence is classified as a match."""),
        html.Div(html.Img(src='data:image/gif;base64,{}'.format(gif_dtw), style={'width': '80%'}), style={'text-align': 'center'})
    ],style={'marginLeft':50})
        
    ])

@webapp.callback(Output("plot_test", "children"), 
              Output("plot_test2", "children"),
              Output("plot_test3", "children"),
              Output("plot_test4", "children"),
              Input("total_c", "clickData"),
              prevent_initial_call=True)

def display_country_plot(feature):
    if feature is not None:
        country_name = feature['properties']['name'] 
        if country_name in hist_df.columns:
            filtered_data = hist_df.loc[:,country_name]
            fig = px.line(x=filtered_data.index, y=filtered_data, title=country_name) 
            fig.update_yaxes(visible=True, fixedrange=True)
            fig.update_layout(annotations=[], overwrite=True)
            fig.update_layout(
                showlegend=False,plot_bgcolor="white",margin=dict(t=60,l=40,b=5,r=5),
                xaxis_title='',yaxis_title='',yaxis=dict(showgrid=True),
                title=dict(text=country_name, font=dict(size=20, color='black'), x=0.5))
            fig.update_traces(line=dict(color='black'))
            filtered_data = pred_df.loc[:,country_name]
            f_min = pred_df_min.loc[:,country_name]
            f_max = pred_df_max.loc[:,country_name]
            fig2 = px.line(x=filtered_data.index, y=filtered_data, title="Mean of Past Future").update_traces(line=dict(color='red'))
            fig2.add_scatter(x =filtered_data.index, y = pred_df_min.loc[:,country_name],
                           mode = 'lines',showlegend=True,opacity=0.2,name='Confidence Interval 95%').update_traces(marker=dict(color='red'))
            fig2.add_scatter(x =filtered_data.index, y = pred_df_max.loc[:,country_name],
                           mode = 'lines',showlegend=True,opacity=0.2,name='Confidence Interval 95%').update_traces(marker=dict(color='red'))
            fig2.update_layout(
                showlegend=False,
                plot_bgcolor="white",
                margin=dict(t=60,l=40,b=5,r=5),
                xaxis_title='',
                yaxis_title='',
                yaxis=dict(showgrid=True),
                title=dict(text="Mean of Past Future", font=dict(size=16, color='darkgrey'), x=0.5))
            m_names = [str(world[world['name_alt']==i.name]['iso_a3'].iloc[0])+': '+str(i.index[0].month)+'-'+str(i.index[0].year) for i in new_dict[country_name][:5]]
            m_dist = dist_dict[country_name]
            if len(m_dist) !=0:
                fig3 = px.bar(x=m_names, y=m_dist, title='Best Matches',
                              color_discrete_sequence=['grey']).update_layout(margin=dict(t=60,l=30,b=5,r=5), 
                              xaxis_title='',yaxis_title='',showlegend=False,plot_bgcolor="white", xaxis_tickangle=60,title=dict(text='Best Matches', font=dict(size=16, color='darkgrey'), x=0.5))
                seq_f=df_d.loc[:,country_name]
                figalp = px.line(x=seq_f.index, y=seq_f, title=country_name) 
                figalp.update_yaxes(visible=True, fixedrange=True)
                figalp.update_layout(annotations=[], overwrite=True)
                figalp.update_layout(
                    showlegend=False,plot_bgcolor="white",margin=dict(t=60,l=40,b=5,r=5),
                    xaxis_title='',yaxis_title='',yaxis=dict(showgrid=True),
                    title=dict(text=f'Decrease pr={df_perc.loc[0,country_name]}%', font=dict(size=20, color='black'), x=0.5))
                figalp.update_traces(line=dict(color='black'))    
                last_six_points = seq_f.tail(6)
                last_sept_points = seq_f.tail(7)
                figalp.add_trace(go.Scatter(
                    x=last_six_points.index,
                    y=last_six_points.values,
                    mode='markers+lines',
                    marker=dict(color='black'),
                    line=dict(color='black'),
                    name='Past Future'
                ))
                figalp.add_trace(go.Scatter(
                    x=last_sept_points.index,
                    y=last_sept_points.values,
                    mode='lines',
                    marker=dict(color='black'),
                    line=dict(color='black')
                ))
                
                
                seq_f=df_s.loc[:,country_name]
                figalp2 = px.line(x=seq_f.index, y=seq_f, title=country_name) 
                figalp2.update_yaxes(visible=True, fixedrange=True)
                figalp2.update_layout(annotations=[], overwrite=True)
                figalp2.update_layout(
                    showlegend=False,plot_bgcolor="white",margin=dict(t=60,l=40,b=5,r=5),
                    xaxis_title='',yaxis_title='',yaxis=dict(showgrid=True),
                    title=dict(text=f'Stable pr={df_perc.loc[1,country_name]}%', font=dict(size=20, color='rgb(216, 134, 141)'), x=0.5))
                figalp2.update_traces(line=dict(color='black'))    
                last_six_points = seq_f.tail(6)
                last_sept_points = seq_f.tail(7)
                figalp2.add_trace(go.Scatter(
                    x=last_six_points.index,
                    y=last_six_points.values,
                    mode='markers+lines',
                    marker=dict(color='rgb(216, 134, 141)'),
                    line=dict(color='rgb(216, 134, 141)'),
                    name='Past Future'
                ))
                figalp2.add_trace(go.Scatter(
                    x=last_sept_points.index,
                    y=last_sept_points.values,
                    mode='lines',
                    marker=dict(color='rgb(216, 134, 141)'),
                    line=dict(color='rgb(216, 134, 141)')
                ))
                
                seq_f=df_i.loc[:,country_name]
                figalp3 = px.line(x=seq_f.index, y=seq_f, title=country_name) 
                figalp3.update_yaxes(visible=True, fixedrange=True)
                figalp3.update_layout(annotations=[], overwrite=True)
                figalp3.update_layout(
                    showlegend=False,plot_bgcolor="white",margin=dict(t=60,l=40,b=5,r=5),
                    xaxis_title='',yaxis_title='',yaxis=dict(showgrid=True),
                    title=dict(text=f'Increase pr={df_perc.loc[2,country_name]}%', font=dict(size=20, color='red'), x=0.5))
                figalp3.update_traces(line=dict(color='black'))    
                last_six_points = seq_f.tail(6)
                last_sept_points = seq_f.tail(7)
                figalp3.add_trace(go.Scatter(
                    x=last_six_points.index,
                    y=last_six_points.values,
                    mode='markers+lines',
                    marker=dict(color='red'),
                    line=dict(color='red'),
                    name='Past Future'
                ))
                figalp3.add_trace(go.Scatter(
                    x=last_sept_points.index,
                    y=last_sept_points.values,
                    mode='lines',
                    marker=dict(color='red'),
                    line=dict(color='red')
                ))
                
                figd = [generate_subplot_sce(series,'black') for series in dict_sce[country_name][0]]
                figs = [generate_subplot_sce(series,'rgb(216, 134, 141)') for series in dict_sce[country_name][1]]
                figi = [generate_subplot_sce(series,'red') for series in dict_sce[country_name][2]]
                # sce = html.Div([
                #     # First row of graphs
                #     html.Div([
                #         dcc.Graph(figure=figalp, style={'width': '33vw', 'height': '30vh'}, config=config),
                #         dcc.Graph(figure=figalp2, style={'width': '33vw', 'height': '30vh'}, config=config),
                #         dcc.Graph(figure=figalp3, style={'width': '33vw', 'height': '30vh'}, config=config)
                #     ], style={"display": "flex", 'flexDirection': 'row', 'marginTop': '3vh'}),
                
                #     # Second row of graphs
                #     html.Div([
                #         html.Div([dcc.Graph(figure=fig, style={'width': '33vw', 'height': '23vh'}, config=config) for fig in figd], style={"display": "flex", 'flexDirection': 'column', 'width': '33vw'}),
                #         html.Div([dcc.Graph(figure=fig, style={'width': '33vw', 'height': '23vh'}, config=config) for fig in figs], style={"display": "flex", 'flexDirection': 'column', 'width': '33vw'}),
                #         html.Div([dcc.Graph(figure=fig, style={'width': '33vw', 'height': '23vh'}, config=config) for fig in figi], style={"display": "flex", 'flexDirection': 'column', 'width': '33vw'})
                #     ], style={"display": "flex", 'flexDirection': 'row'})
                # ])   
                sce = html.Div([
                    # First row of graphs
                    html.Div([
                        dcc.Graph(figure=figalp, className='sub-graph-sce', config=config),
                        html.Div([dcc.Graph(figure=fig, className='sub-graph-sce', config=config) for fig in figd], style={"display": "flex", 'flexDirection': 'column'})
                    ], style={ 'marginTop': '3vh'}),
                    html.Div([
                        dcc.Graph(figure=figalp2, className='sub-graph-sce', config=config),
                        html.Div([dcc.Graph(figure=fig, className='sub-graph-sce', config=config) for fig in figs], style={"display": "flex", 'flexDirection': 'column'})
                    ], style={'marginTop': '3vh'}),
                    html.Div([
                        dcc.Graph(figure=figalp3, className='sub-graph-sce', config=config),
                        html.Div([dcc.Graph(figure=fig, className='sub-graph-sce', config=config) for fig in figi], style={"display": "flex", 'flexDirection': 'column'})
                    ], style={ 'marginTop': '3vh'}),
                ],className='graph-container-sce')   
            else:
                fig3 = px.bar().update_layout(margin=dict(t=30,l=30,b=5,r=5), 
                xaxis_title='',yaxis_title='',showlegend=False,plot_bgcolor="white",title=dict(text='Best Matches', font=dict(size=16, color='darkgrey'), x=0.5))        
                fig3.update_xaxes(showticklabels=False)
                fig3.update_yaxes(showticklabels=False)  
                sce=[]                                   
            figs = [generate_subplot(series) for series in new_dict[country_name]]
            rows = []
            for i in range(0, len(figs), 3):
                row = html.Div([dcc.Graph(figure=fig,className='sub-graph',config=config) for fig in figs[i:i+3]], className='graph-container',style={"display": "flex"})
                rows.append(row)


            
            return (dcc.Graph(figure=fig,className='plot-emb',config=config),
                    dcc.Graph(figure=fig2,className='plot-emb',config=config),
                    dcc.Graph(figure=fig3,className='plot-emb',config=config),
                    dcc.Tabs([dcc.Tab(label='Matched Sequences',children=rows,value='tab-1'),dcc.Tab(label='Scenarios',children=sce,value='tab-2')],colors={"border": "#555", 'background': '#D3D3D3'}))


@webapp.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/about':
        return about_layout
    elif pathname == '/report':
        return report_layout
    else:
        return home_layout
    

webapp.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

if __name__ == '__main__':
    webapp.run_server()
    