# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:24:00 2023

@author: thoma
"""

from tslearn.clustering import TimeSeriesKMeans
from scipy.spatial.distance import euclidean
import pandas as pd
from shape import Shape,finder
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import seaborn as sns
import pickle
from datetime import datetime,date,timedelta
import os 

df_tot_m = pd.read_csv('Conf.csv',parse_dates=True,index_col=(0))

df_next = {0:pd.DataFrame(columns=df_tot_m.columns,index=range(16)),1:pd.DataFrame(columns=df_tot_m.columns,index=range(16)),2:pd.DataFrame(columns=df_tot_m.columns,index=range(16))}
df_perc = pd.DataFrame(columns=df_tot_m.columns,index=range(3))

dict_sce = {i :[[],[],[]] for i in df_tot_m.columns}

h_train=10
h=6
#clu_thr=2

for coun in range(len(df_tot_m.columns)):
    if not (df_tot_m.iloc[-h_train:,coun]==0).all():
        pred_tot=[]
        shape = Shape()
        shape.set_shape(df_tot_m.iloc[-h_train:,coun]) 
        find = finder(df_tot_m.iloc[:-h,:],shape)
        find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
        min_d_d=0.05
        while len(find.sequences)<3:
            min_d_d += 0.05
            find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=2)
        pred_ori = find.predict(horizon=h,plot=False,mode='mean')
        seq_pred =find.predict(horizon=h,plot=False,mode='mean',seq_out=True)
        
        # X = seq_pred.values
        # n_clusters = 3
        # model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
        # y_pred = model.fit_predict(X)
        # centroids = model.cluster_centers_
        # cluster_distances = []
        # for i in range(n_clusters):
        #     for j in range(i + 1, n_clusters):
        #         cluster_distances.append(euclidean(centroids[i].ravel(), centroids[j].ravel()))
        # if min(cluster_distances)<clu_thr:
        #     n_clusters = 2
        #     model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
        #     y_pred = model.fit_predict(X)
        #     centroids = model.cluster_centers_
        #     cluster_distances = []
        #     for i in range(n_clusters):
        #         for j in range(i + 1, n_clusters):
        #             cluster_distances.append(euclidean(centroids[i].ravel(), centroids[j].ravel()))
        #     if min(cluster_distances) < clu_thr:
        #         n_clusters=1
        #         y_pred = np.array([0]*len(y_pred))
        
        y_pred = np.full((len(seq_pred),),np.nan)
        y_pred[seq_pred.sum(axis=1)<=1.5] = 0
        y_pred[(seq_pred.sum(axis=1)>1.5) & (seq_pred.sum(axis=1)<5)] = 1
        y_pred[seq_pred.sum(axis=1)>=5] = 2
        
        perc=[]
        for i in range(3):#len(pd.Series(y_pred).value_counts())):
            if len(seq_pred[y_pred==i]) != 0:
                norm = seq_pred[y_pred==i].mean() * (df_tot_m.iloc[-h_train:,coun].max()-df_tot_m.iloc[-h_train:,coun].min()) + df_tot_m.iloc[-h_train:,coun].min()
                norm.index = pd.date_range(start=df_tot_m.iloc[-h_train:,coun].index[-1] + pd.DateOffset(months=1), periods=6, freq='M')
                seq_f = pd.concat([df_tot_m.iloc[-h_train:,coun],norm],axis=0)
                index_s = seq_f.index
                seq_f = seq_f.reset_index(drop=True)
                df_next[i].iloc[:,coun] = seq_f
            else : 
                df_next[i].iloc[:,coun] = [float('nan')]*16
            perc.append(round(len(seq_pred[y_pred==i])/len(seq_pred)*100,1))
        df_perc.iloc[:,coun]=perc
        
        y_pred_p = y_pred.astype(float)
        count_zeros = 0
        count_ones = 0
        count_twos = 0
        for i in range(len(y_pred_p)):
            value = y_pred_p[i]
            norm = seq_pred.iloc[i,:]*(find.sequences[i][0].max()-find.sequences[i][0].min())+ find.sequences[i][0].min()
            norm.index = pd.date_range(start=find.sequences[i][0].index[-1] + pd.DateOffset(months=1), periods=6, freq='M')
            seq_f = pd.concat([find.sequences[i][0],norm],axis=0)
            seq_f.name = find.sequences[i][0].name
            if value == 0 and count_zeros < 5:   
                dict_sce[df_tot_m.columns[coun]][0].append(seq_f)
                count_zeros += 1
            elif value == 1 and count_ones < 5:
                dict_sce[df_tot_m.columns[coun]][1].append(seq_f)
                count_ones += 1
            elif value == 2 and count_twos < 5:
                dict_sce[df_tot_m.columns[coun]][2].append(seq_f)
                count_twos += 1
            else:
                pass
        

        
        # l_colo = ['#E41B17','blue','orange']
        # plt.figure(figsize=(15, 6))
        # for i in y_pred_p.index:
        #     norm = (find.sequences[i][0] - find.sequences[i][0].min()) / (find.sequences[i][0].max()-find.sequences[i][0].min())
        #     norm.index = range(12-len(norm),12)
        #     pre = pd.Series([norm.iloc[-1]]+seq_pred.iloc[i,:].tolist())
        #     pre.index = range(11,18)
        #     plt.plot(norm,alpha=0.5,color='black')
        #     plt.plot(pre,alpha=0.3,color=l_colo[int(y_pred_p[i])])
        # maxi_v=0    
        # for i in range(3): 
        #     pre = pd.Series([shape.values[-1]]+mean_l[i].tolist())
        #     pre.index = range(11,18)    
        #     plt.plot(pre,marker='o',alpha=1,linewidth=5,color=l_colo[i],label=f'Scenario {i} {perc[i]}%')
        #     if max(pre)>maxi_v:
        #         maxi_v=max(pre)
        # if maxi_v>5:
        #     plt.ylim(0,5.5)
        # elif (maxi_v>1) and (maxi_v<=5):
        #     plt.ylim(0,maxi_v+0.1*maxi_v)
        # else:
        #     plt.ylim(0,1)
        # plt.xticks([*range(18)],[f't-{i}' for i in range(1,12)][::-1]+['t']+[f't+{i}' for i in range(1,7)])
        # plt.xticks([*range(18)],['']*12+[f't+{i}' for i in range(1,7)])
        # plt.box(False)
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.xticks(fontsize=20)  # Set x-axis tick font size
        # plt.yticks(fontsize=20)
        # plt.legend(fontsize=15)
        # plt.title(df_tot_m.columns[coun])
        # plt.show()

df_next[0].index = index_s
df_next[1].index = index_s
df_next[2].index = index_s

def rena_f(df):
    df = df.rename(columns={'Bosnia-Herzegovina':'Bosnia and Herz.','Cambodia (Kampuchea)':'Cambodia',
                                    'Central African Republic':'Central African Rep.','DR Congo (Zaire)':'Dem. Rep. Congo',
                                    'Ivory Coast':'Côte d\'Ivoire','Kingdom of eSwatini (Swaziland)':'eSwatini', 'Dominican Republic':'Dominican Rep.',
                                    'Macedonia, FYR':'Macedonia','Madagascar (Malagasy)':'Madagascar','Myanmar (Burma)':'Myanmar',
                                    'Russia (Soviet Union)':'Russia','Serbia (Yugoslavia)':'Serbia','South Sudan':'S. Sudan',
                                    'Yemen (North Yemen)':'Yemen','Zimbabwe (Rhodesia)':'Zimbabwe','Vietnam (North Vietnam)':'Vietnam'})
    return df

df_perc=rena_f(df_perc)
df_next[0]=rena_f(df_next[0])
df_next[1]=rena_f(df_next[1])
df_next[2]=rena_f(df_next[2])

df_perc.to_csv('perc.csv')
df_next[0].to_csv('dec.csv')
df_next[1].to_csv('sta.csv')
df_next[2].to_csv('inc.csv')

rena={'Bosnia-Herzegovina':'Bosnia and Herz.','Cambodia (Kampuchea)':'Cambodia',
                                   'Central African Republic':'Central African Rep.','DR Congo (Zaire)':'Dem. Rep. Congo',
                                   'Ivory Coast':'Côte d\'Ivoire','Kingdom of eSwatini (Swaziland)':'eSwatini', 'Dominican Republic':'Dominican Rep.',
                                   'Macedonia, FYR':'Macedonia','Madagascar (Malagasy)':'Madagascar','Myanmar (Burma)':'Myanmar',
                                   'Russia (Soviet Union)':'Russia','Serbia (Yugoslavia)':'Serbia','South Sudan':'S. Sudan',
                                   'Yemen (North Yemen)':'Yemen','Zimbabwe (Rhodesia)':'Zimbabwe','Vietnam (North Vietnam)':'Vietnam'}
dict_sce_f = {rena[key] if key in rena else key: item for key, item in dict_sce.items()}
with open('dict_sce.pkl', 'wb') as f:
    pickle.dump(dict_sce_f, f)
