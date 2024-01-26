# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:40:19 2023

@author: thoma
"""


import matplotlib.pyplot as plt
import json
import pandas as pd
import pickle 
import base64
from PIL import Image
import tweepy
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from dateutil.relativedelta import relativedelta
from datetime import datetime,date
from matplotlib.font_manager import FontProperties

# Register the Poppins font
font_path = 'Poppins/Poppins-Regular.ttf'
prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

with open('saved_dictionary.pkl', 'rb') as f:
    dict_m_o = pickle.load(f)
rena={'Bosnia-Herzegovina':'Bosnia and Herz.','Cambodia (Kampuchea)':'Cambodia',
                                   'Central African Republic':'Central African Rep.','DR Congo (Zaire)':'Dem. Rep. Congo',
                                   'Ivory Coast':'Côte d\'Ivoire','Kingdom of eSwatini (Swaziland)':'eSwatini',
                                   'Macedonia, FYR':'Macedonia','Madagascar (Malagasy)':'Madagascar','Myanmar (Burma)':'Myanmar',
                                   'Russia (Soviet Union)':'Russia','Serbia (Yugoslavia)':'Serbia','South Sudan':'S. Sudan',
                                   'Yemen (North Yemen)':'Yemen','Zimbabwe (Rhodesia)':'Zimbabwe','Vietnam (North Vietnam)':'Vietnam'}
dict_m = {rena[key] if key in rena else key: item for key, item in dict_m_o.items()}

df_tot_m = pd.read_csv('Conf.csv',parse_dates=True,index_col=(0))
df_tot_m = df_tot_m.rename(columns={'Bosnia-Herzegovina':'Bosnia and Herz.','Cambodia (Kampuchea)':'Cambodia',
                                    'Central African Republic':'Central African Rep.','DR Congo (Zaire)':'Dem. Rep. Congo',
                                    'Ivory Coast':'Côte d\'Ivoire','Kingdom of eSwatini (Swaziland)':'eSwatini',
                                    'Macedonia, FYR':'Macedonia','Madagascar (Malagasy)':'Madagascar','Myanmar (Burma)':'Myanmar',
                                    'Russia (Soviet Union)':'Russia','Serbia (Yugoslavia)':'Serbia','South Sudan':'S. Sudan',
                                    'Yemen (North Yemen)':'Yemen','Zimbabwe (Rhodesia)':'Zimbabwe','Vietnam (North Vietnam)':'Vietnam'})

hist_df=pd.read_csv('Hist.csv',parse_dates=True,index_col=(0))
column_means = hist_df.mean()
threshold = 0.4
df_filtered = hist_df.loc[:, (hist_df > column_means).mean() > threshold]
min_d_list=[]
for i in df_filtered.columns:
    min_d_list.append(dict_m[i][0][1])    
min_d_list = pd.Series(min_d_list,index=df_filtered.columns)
min_d_list = min_d_list.sort_values()

for i in range(10):
    logo_path = 'Images/PaCE_final.png'  # Replace with the actual path to your logo or image
    logo_img = plt.imread(logo_path)
    imagebox = OffsetImage(logo_img, zoom=0.2)  # You can adjust the zoom factor as needed
    ab = AnnotationBbox(imagebox, (0.9, 1), frameon=False, xycoords='axes fraction', boxcoords="axes fraction")
    sub_name= dict_m[min_d_list.index[i]][0][0].name
    if sub_name in rena:
        sub_name = rena[sub_name]
    fig, ax = plt.subplot_mosaic([[0,0,0,2,2],[1,1,1,1,1]], figsize=(15, 12))
    ax[0].plot(hist_df.loc[:,min_d_list.index[i]], marker='o', color='black', linestyle='-', linewidth=2, markersize=8)
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    ax[0].set_xlabel('Date', fontsize=20, font='Poppins')
    ax[0].tick_params(axis='x', labelsize=16, rotation=45)
    ax[0].tick_params(axis='y', labelsize=16)
    ax[0].set_title(hist_df.loc[:,min_d_list.index[i]].name, fontsize=25, font='Poppins')
    ax[0].set_frame_on(False)
    
    ax[1].plot(df_tot_m.loc[dict_m[min_d_list.index[i]][0][0].index[-1]:,sub_name].iloc[:7], marker='o', color='#df2226', linestyle='-', linewidth=2, markersize=8)
    ax[1].plot(dict_m[min_d_list.index[i]][0][0], marker='o', color='gray', linestyle='-', linewidth=2, markersize=8)
    ax[1].grid(axis='y', linestyle='--', alpha=0.7)
    ax[1].set_xlabel('Date', fontsize=20, color='gray', font='Poppins')
    ax[1].set_title(sub_name, fontsize=25, color='gray', font='Poppins')
    ax[1].set_frame_on(False)
    ax[1].text(x=df_tot_m.loc[dict_m[min_d_list.index[i]][0][0].index[-1]:,sub_name].iloc[:7].index[0],y=plt.ylim()[0] - 4*(plt.ylim()[1] - plt.ylim()[0]) / 10,s='Source : UCDP (https://ucdp.uu.se/downloads/)',fontsize=15,color='darkgrey')
    ax[1].tick_params(axis='x', labelsize=16, rotation=45, labelcolor='gray')
    ax[1].tick_params(axis='y', labelsize=16, labelcolor='gray')
    
    ax[2].add_artist(ab)
    lines = ['Actual','Best historical match','Future casualties for the match']
    text_objects = [
        ax[2].text(0.1, 0.4 - i * 0.1, line, fontsize=20, font='Poppins', color=color, ha='left', va='baseline', linespacing=1.5)
        for i, line, color in zip(range(len(lines)), lines, ['black', 'gray', '#df2226'])
    ]
    ax[2].text(0.4, 0.7, 'Conflict Fatalities', fontsize=30, color='black', ha='center', weight='bold', font='Poppins')
    ax[2].axis('off')
    plt.tight_layout()
    plt.savefig(f'Images\{i}_c.png', bbox_inches='tight')
    plt.show()

    
    
    

# API_key ='iKE87AGJhYpj2dlZpBba57Ohf'
# key_secret ='w3RYtCA3kQJc0wImvEYWTQu5iTY9MrA0riVBceehAVT5Zomvy6'
# access_token='1618319345720909838-axd6L7hd3q9f7ISYQ4kHrb7mH2Btw1'
# secret_token='JlaVU540DRUSPbarc70SFj1EzxTpjJ2QmmAwyFK3dhRzE'

# client_id = 'TU54MnlxRG9ySV9xajRvaDRDbWI6MTpjaQ'
# client_sec='f9BlnK1v6AHAl198Ap4DYzv8NQo38G-Vd5M8wSSvUQk6xfUNgp'
# bearer_tok='AAAAAAAAAAAAAAAAAAAAAHKJrQEAAAAA16QgTBO%2B1Fq10JsVkzWFvVgTNMw%3D6s601sGIPqThS2pQMwMhvJHDcBRNL2rlbAVNUKJvxntyAae7cv'

# auth = tweepy.OAuth1UserHandler(API_key, key_secret)
# auth.set_access_token(access_token, secret_token)

# api = tweepy.API(auth)
# media = api.media_upload('github-mark.png')
# media_id = media.media_id


# c_1='Test1'
# c_2='Test2'
# mon=datetime.now().strftime("%B")
# ye= datetime.now().strftime("%Y")
# message = f"We're observing a striking resemblance in conflict fatalities patterns: Current trends in Country @{c_1} (up to this month) closely mirror those of @{c_2} starting in @{mon}-@{ye}. The red line shows the unfolding events in @{c_2}."
# client = tweepy.Client(bearer_token=bearer_tok,consumer_key=API_key,consumer_secret=key_secret,access_token=access_token,access_token_secret=secret_token)
# client.create_tweet(text=message, media_ids=[media_id])



