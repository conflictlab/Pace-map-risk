# -*- coding: utf-8 -*-

"""
Created on Mon Jan 15 22:24:43 2024

@author: thoma
"""


from datetime import datetime,date
from dateutil.relativedelta import relativedelta
import pandas as pd
import pickle 
import geopandas as gpd

world = gpd.read_file('world_plot.geojson')
pred_df=pd.read_csv('Pred_df.csv',parse_dates=True,index_col=(0))
missing_columns = set(world['name']) - set(pred_df.columns)
with open('sce_dictionary.pkl', 'rb') as f:
    dict_sce = pickle.load(f)
missing_columns.discard('Dominican Rep.')    
dict_sce.update({col: [[],[]] for col in missing_columns})
hist_df=pd.read_csv('Hist.csv',parse_dates=True,index_col=(0))
hist_df = hist_df.assign(**{col: [0] * 10 for col in missing_columns})

l_pred=[]
na=[]
for i,j in enumerate(dict_sce.keys()):
    sub = dict_sce[j][1]
    if len(sub)==0:
        na.append(j)
        l_pred.append(pd.Series([0]*6))
    else:
        if type(sub.loc[sub.index.max(),:]) == pd.Series:
            sub_p=sub.loc[sub.index.max(),:]
            sub_p = sub_p * (hist_df.loc[:,j].max() - hist_df.loc[:,j].min()) + hist_df.loc[:,j].min()
            na.append(j)
            l_pred.append(sub_p)
        else:
            sub_p=pd.DataFrame(sub.loc[sub.index.max(),:]).mean(axis=0)
            sub_p = sub_p * (hist_df.loc[:,j].max() - hist_df.loc[:,j].min()) + hist_df.loc[:,j].min()
            na.append(j)
            l_pred.append(sub_p)
        
l_pred = [series.reset_index(drop=True) for series in l_pred]
df_pred = pd.concat(l_pred, axis=1)
df_pred.columns = na
df_pred.index = sub_p.index

month = datetime.now().strftime("%b")
year = datetime.now().strftime("%Y")
until = date.today() + relativedelta(months=-1)
month_t = until.strftime("%Y")
s_month_t = until.strftime("%b")
month_t_l = until.strftime("%B")
six_months = date.today() + relativedelta(months=+5)
sm_m = six_months.strftime("%b")
sm_y = six_months.strftime("%Y")

df_pred.to_csv(f'Historical_Predictions/{month_t_l}-{month_t}_{month}-{year}_to_{sm_m}-{sm_y}.csv')



