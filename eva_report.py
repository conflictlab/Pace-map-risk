# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:01:53 2024

@author: thoma
"""

import pandas as pd
from shape import Shape,finder
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import geopandas as gpd
import seaborn as sns
import pickle
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from datetime import datetime,timedelta
import math
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
from PIL import Image


df = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged231-csv.zip",
                  parse_dates=['date_start','date_end'],low_memory=False)
df= pd.concat([df,pd.read_csv('https://ucdp.uu.se/downloads/candidateged/GEDEvent_v23_01_23_12.csv',parse_dates=['date_start','date_end'],low_memory=False)],axis=0)
df_tot = pd.DataFrame(columns=df.country.unique(),index=pd.date_range(df.date_start.min(),
                                          df.date_end.max()))
df_tot=df_tot.fillna(0)
for i in df.country.unique():
    df_sub=df[df.country==i]
    for j in range(len(df_sub)):
        if df_sub.date_start.iloc[j].month == df_sub.date_end.iloc[j].month:
            df_tot.loc[df_sub.date_start.iloc[j],i]=df_tot.loc[df_sub.date_start.iloc[j],i]+df_sub.best.iloc[j]
        else:
            pass                                                    
                                                     
df_tot_m=df_tot.resample('M').sum()
del df
del df_tot
df_tot_m=df_tot_m.iloc[:-1,:]

h_train=10
h=6
pred_tot=[]
dict_m={i :[] for i in df_tot_m.columns}
for coun in range(len(df_tot_m.columns)):
    if not (df_tot_m.iloc[-h_train:,coun]==0).all():
        shape = Shape()
        shape.set_shape(df_tot_m.iloc[-h_train:,coun]) 
        find = finder(df_tot_m.iloc[:-h,:],shape)
        find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2,min_mat=3,d_increase=0.05)
        find.create_sce_predict(h)
        clu = find.val_sce
        dict_m[df_tot_m.columns[coun]] = clu
        pr = clu.index
        pred_ori=clu.iloc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_tot_m.iloc[-h_train:,coun].max()-df_tot_m.iloc[-h_train:,coun].min())+df_tot_m.iloc[-h_train:,coun].min()
        pred_tot.append(preds)
    else :
        pred_tot.append(pd.Series(np.zeros((h,))))

df_pred = pd.concat(pred_tot,axis=1)        
df_pred.columns = df_tot_m.columns

df = pd.read_csv("https://ucdp.uu.se/downloads/candidateged/GEDEvent_v24_01_24_03.csv",
                  parse_dates=['date_start','date_end'],low_memory=False)
df= pd.concat([df,pd.read_csv('https://ucdp.uu.se/downloads/candidateged/GEDEvent_v24_0_4.csv',parse_dates=['date_start','date_end'],low_memory=False)],axis=0)
df_tot = pd.DataFrame(columns=df.country.unique(),index=pd.date_range(df.date_start.min(),
                                          df.date_end.max()))
df_tot=df_tot.fillna(0)
for i in df.country.unique():
    df_sub=df[df.country==i]
    for j in range(len(df_sub)):
        if df_sub.date_start.iloc[j].month == df_sub.date_end.iloc[j].month:
            df_tot.loc[df_sub.date_start.iloc[j],i]=df_tot.loc[df_sub.date_start.iloc[j],i]+df_sub.best.iloc[j]
        else:
            pass                                                    
df_tot_true=df_tot.resample('M').sum()                    
df_tot_true=df_tot_true.iloc[1:-1]

mse_l=[]
for coun in df_tot_m.columns:
    if coun in df_tot_true.columns:
        plt.figure(figsize=(12,6))
        plt.plot(df_tot_m.iloc[-4:].loc[:,coun],color='black')
        pre = df_pred.loc[:3,coun]
        pre.index = df_tot_true.loc[:,coun].index
        plt.plot(pre,label='Pred')
        plt.plot(df_tot_true.loc[:,coun],label='True')
        plt.legend()
        plt.show()

