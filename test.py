# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:31:17 2023

@author: thoma
"""

import pandas as pd
from shape import Shape,finder
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import seaborn as sns
import pickle
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from datetime import datetime,timedelta
import math

df = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged231-csv.zip",
                  parse_dates=['date_start','date_end'],low_memory=False)
df= pd.concat([df,pd.read_csv('https://ucdp.uu.se/downloads/candidateged/GEDEvent_v23_01_23_12.csv',parse_dates=['date_start','date_end'],low_memory=False)],axis=0)
month = datetime.now().strftime("%m")
for i in range(1,int(month)):
    df= pd.concat([df,pd.read_csv(f'https://ucdp.uu.se/downloads/candidateged/GEDEvent_v24_0_{i}.csv',parse_dates=['date_start','date_end'],low_memory=False)],axis=0)

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
last_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
df_tot_m= df_tot_m.loc[:last_month,:]
df_tot_m.to_csv('Conf.csv')
del df
del df_tot
#df_tot_m = pd.read_csv('Conf.csv',parse_dates=True,index_col=(0))

df_next = {0:pd.DataFrame(columns=df_tot_m.columns,index=range(16)),1:pd.DataFrame(columns=df_tot_m.columns,index=range(16)),2:pd.DataFrame(columns=df_tot_m.columns,index=range(16))}
df_perc = pd.DataFrame(columns=df_tot_m.columns,index=range(3))
dict_sce = {i :[[],[],[]] for i in df_tot_m.columns}


h_train=10
h=6
pred_tot=[]
pred_raw=[]
dict_m={i :[] for i in df_tot_m.columns}
for coun in range(len(df_tot_m.columns)):
    if not (df_tot_m.iloc[-h_train:,coun]==0).all():
        shape = Shape()
        shape.set_shape(df_tot_m.iloc[-h_train:,coun]) 
        find = finder(df_tot_m.iloc[:-h,:],shape)
        find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
        min_d_d=0.1
        while len(find.sequences)<3:
            min_d_d += 0.05
            find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=2)
        pred_ori = find.predict(horizon=h,plot=False,mode='mean')
        pred_raw.append(pred_ori)
        pred_ori = pred_ori*(df_tot_m.iloc[-h_train:,coun].max()-df_tot_m.iloc[-h_train:,coun].min())+df_tot_m.iloc[-h_train:,coun].min()
        pred_tot.append(pred_ori)
        dict_m[df_tot_m.columns[coun]]=find.sequences
        seq_pred =find.predict(horizon=h,plot=False,mode='mean',seq_out=True)
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
            if value == 0 and count_zeros<5:   
                dict_sce[df_tot_m.columns[coun]][0].append(seq_f)
                count_zeros += 1
            elif value == 1 and count_ones<5:
                dict_sce[df_tot_m.columns[coun]][1].append(seq_f)
                count_ones += 1
            elif value == 2 and count_twos<5:
                dict_sce[df_tot_m.columns[coun]][2].append(seq_f)
                count_twos += 1
            else:
                pass
    else :
        pred_tot.append(pd.DataFrame(np.zeros((h,3))))
        pred_raw.append(pd.DataFrame(np.zeros((h,3))))
        
with open('saved_dictionary.pkl', 'wb') as f:
    pickle.dump(dict_m, f)
    
pred_df = [df.iloc[:, 0] for df in pred_raw]
pred_df = pd.concat(pred_df, axis=1)
pred_df.columns = df_tot_m.columns
pred_df = pred_df.rename(columns={'Bosnia-Herzegovina':'Bosnia and Herz.','Cambodia (Kampuchea)':'Cambodia',
                                    'Central African Republic':'Central African Rep.','DR Congo (Zaire)':'Dem. Rep. Congo',
                                    'Ivory Coast':'Côte d\'Ivoire','Kingdom of eSwatini (Swaziland)':'eSwatini', 'Dominican Republic':'Dominican Rep.',
                                    'Macedonia, FYR':'Macedonia','Madagascar (Malagasy)':'Madagascar','Myanmar (Burma)':'Myanmar',
                                    'Russia (Soviet Union)':'Russia','Serbia (Yugoslavia)':'Serbia','South Sudan':'S. Sudan',
                                    'Yemen (North Yemen)':'Yemen','Zimbabwe (Rhodesia)':'Zimbabwe','Vietnam (North Vietnam)':'Vietnam'})

pred_df.to_csv('Pred_df.csv')
pred_df_m = [df.iloc[:, 1] for df in pred_raw]
pred_df_m = pd.concat(pred_df_m, axis=1)
pred_df_m.columns = df_tot_m.columns
pred_df_m = pred_df_m.rename(columns={'Bosnia-Herzegovina':'Bosnia and Herz.','Cambodia (Kampuchea)':'Cambodia',
                                    'Central African Republic':'Central African Rep.','DR Congo (Zaire)':'Dem. Rep. Congo',
                                    'Ivory Coast':'Côte d\'Ivoire','Kingdom of eSwatini (Swaziland)':'eSwatini', 'Dominican Republic':'Dominican Rep.',
                                    'Macedonia, FYR':'Macedonia','Madagascar (Malagasy)':'Madagascar','Myanmar (Burma)':'Myanmar',
                                    'Russia (Soviet Union)':'Russia','Serbia (Yugoslavia)':'Serbia','South Sudan':'S. Sudan',
                                    'Yemen (North Yemen)':'Yemen','Zimbabwe (Rhodesia)':'Zimbabwe','Vietnam (North Vietnam)':'Vietnam'})

pred_df_m.to_csv('Pred_df_min.csv')
pred_df_m = [df.iloc[:, 2] for df in pred_raw]
pred_df_m = pd.concat(pred_df_m, axis=1)
pred_df_m.columns = df_tot_m.columns
pred_df_m = pred_df_m.rename(columns={'Bosnia-Herzegovina':'Bosnia and Herz.','Cambodia (Kampuchea)':'Cambodia', 'Dominican Republic':'Dominican Rep.',
                                    'Central African Republic':'Central African Rep.','DR Congo (Zaire)':'Dem. Rep. Congo',
                                    'Ivory Coast':'Côte d\'Ivoire','Kingdom of eSwatini (Swaziland)':'eSwatini',
                                    'Macedonia, FYR':'Macedonia','Madagascar (Malagasy)':'Madagascar','Myanmar (Burma)':'Myanmar',
                                    'Russia (Soviet Union)':'Russia','Serbia (Yugoslavia)':'Serbia','South Sudan':'S. Sudan',
                                    'Yemen (North Yemen)':'Yemen','Zimbabwe (Rhodesia)':'Zimbabwe','Vietnam (North Vietnam)':'Vietnam'})

pred_df_m.to_csv('Pred_df_max.csv')


#pred_df=pd.read_csv('Pred_df.csv',parse_dates=True,index_col=(0))
pred_df = [df.iloc[:, 0] for df in pred_tot]
pred_df = pd.concat(pred_df, axis=1)
pred_df.columns = df_tot_m.columns
pred_df = pred_df.rename(columns={'Bosnia-Herzegovina':'Bosnia and Herz.','Cambodia (Kampuchea)':'Cambodia',
                                    'Central African Republic':'Central African Rep.','DR Congo (Zaire)':'Dem. Rep. Congo',
                                    'Ivory Coast':'Côte d\'Ivoire','Kingdom of eSwatini (Swaziland)':'eSwatini', 'Dominican Republic':'Dominican Rep.',
                                    'Macedonia, FYR':'Macedonia','Madagascar (Malagasy)':'Madagascar','Myanmar (Burma)':'Myanmar',
                                    'Russia (Soviet Union)':'Russia','Serbia (Yugoslavia)':'Serbia','South Sudan':'S. Sudan',
                                    'Yemen (North Yemen)':'Yemen','Zimbabwe (Rhodesia)':'Zimbabwe','Vietnam (North Vietnam)':'Vietnam'})

value_pred = pred_df.sum().reset_index()
value_pred.columns=['name','value']

histo = df_tot_m.iloc[-h:,:]
histo = histo.rename(columns={'Bosnia-Herzegovina':'Bosnia and Herz.','Cambodia (Kampuchea)':'Cambodia',
                                   'Central African Republic':'Central African Rep.','DR Congo (Zaire)':'Dem. Rep. Congo',
                                   'Ivory Coast':'Côte d\'Ivoire','Kingdom of eSwatini (Swaziland)':'eSwatini', 'Dominican Republic':'Dominican Rep.',
                                   'Macedonia, FYR':'Macedonia','Madagascar (Malagasy)':'Madagascar','Myanmar (Burma)':'Myanmar',
                                   'Russia (Soviet Union)':'Russia','Serbia (Yugoslavia)':'Serbia','South Sudan':'S. Sudan',
                                   'Yemen (North Yemen)':'Yemen','Zimbabwe (Rhodesia)':'Zimbabwe','Vietnam (North Vietnam)':'Vietnam'})
histo = histo.sum().reset_index()
histo.columns=['name','hist']


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.merge(value_pred, how='left',on='name')
world = world.merge(histo, how='left',on='name')
world = world[world.name != 'Antarctica']
world = world.fillna(0)
world.loc[world['value'] < 0, 'value'] = 0
world['per_pred']=world['value']/world['pop_est']
world['log_per_pred']=np.log10(world['value']+1)
world.to_file('world_plot.geojson', driver='GeoJSON') 


df_tot_m_plot=df_tot_m.iloc[-h_train:,:]
df_tot_m_plot = df_tot_m_plot.rename(columns={'Bosnia-Herzegovina':'Bosnia and Herz.','Cambodia (Kampuchea)':'Cambodia',
                                   'Central African Republic':'Central African Rep.','DR Congo (Zaire)':'Dem. Rep. Congo',
                                   'Ivory Coast':'Côte d\'Ivoire','Kingdom of eSwatini (Swaziland)':'eSwatini', 'Dominican Republic':'Dominican Rep.',
                                   'Macedonia, FYR':'Macedonia','Madagascar (Malagasy)':'Madagascar','Myanmar (Burma)':'Myanmar',
                                   'Russia (Soviet Union)':'Russia','Serbia (Yugoslavia)':'Serbia','South Sudan':'S. Sudan',
                                   'Yemen (North Yemen)':'Yemen','Zimbabwe (Rhodesia)':'Zimbabwe','Vietnam (North Vietnam)':'Vietnam'})
df_tot_m_plot.to_csv('Hist.csv')

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


# =============================================================================
# Global Map
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(30, 15))
world.boundary.plot(ax=ax, color='black')
norm = mcolors.Normalize(vmin=0, vmax=math.ceil(max(world['log_per_pred'])))
mapping = world.plot(column='log_per_pred', cmap='Reds', ax=ax, norm=norm)
plt.xlim(-180, 180)
plt.box(False)
ax.spines['left'].set_visible(False)
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_xticks([])
cbar_ax = fig.add_axes([0.65, 0.15, 0.3, 0.02]) 
sm = ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([]) 
cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_ticks([*range(math.ceil(max(world['log_per_pred']))+1)])
cbar.set_ticklabels(['1']+[f'$10^{e}$' for e in range(1,math.ceil(max(world['log_per_pred']))+1)])
plt.text(1.9,1.5,'Risk index', fontsize=30)
plt.text(-8.5,0.1,'The risk index corresponds to the log sum of predicted fatalities in the next 6 months.',color='dimgray', fontdict={'style': 'italic'})
plt.savefig('Images/map.png', bbox_inches='tight')
plt.show()

# =============================================================================
# Historical Plot
# =============================================================================
pred_df.index = pd.date_range(start=df_tot_m.index[-1],periods=h+1,freq='M')[1:]
historical_series = pd.concat([df_tot_m.sum(axis=1).iloc[-60:],pred_df.sum(axis=1)],axis=0)
date_rng = historical_series.index

plt.figure(figsize=(25, 6))
plt.plot(date_rng[:-h+1], historical_series[:-h+1], marker='o', color='grey', linestyle='-', linewidth=2, markersize=8)
plt.plot(date_rng[-h:], historical_series[-h:], marker='o', color='red', linestyle='-', linewidth=2, markersize=8)
plt.scatter(date_rng[-h:], historical_series[-h:], color='red', s=100, zorder=5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel('Date', fontsize=20)
plt.xticks(fontsize=16)  # Set x-axis tick font size
plt.yticks(fontsize=16)
plt.box(False)
plt.xticks(rotation=45, ha='right')
plt.savefig('Images/sub1_1.png', bbox_inches='tight')
plt.show()

# =============================================================================
# Per continent
# =============================================================================

pred_cont = world.groupby('continent').sum()['value']
hist_cont = world.groupby('continent').sum()['hist'] 

df_plot = pd.DataFrame({'pred_cont': pred_cont, 'hist_cont': hist_cont})
df_plot = df_plot[df_plot.index!='Seven seas (open ocean)']
df_plot = df_plot.sort_values('pred_cont')
df_plot['color'] = np.where(df_plot['pred_cont'] > df_plot['hist_cont'], 'red', 'black')
def calculate_alpha(row):
    diff_ratio = (row['pred_cont'] - row['hist_cont']) / row['hist_cont']
    return np.clip(diff_ratio / 2 +0.5 , 0, 1)
df_plot['alpha'] = df_plot.apply(calculate_alpha, axis=1)
plt.figure(figsize=(10, 6))
ax=sns.barplot(x=df_plot.index, y='pred_cont', data=df_plot, palette=df_plot['color'])
for i, bar in enumerate(ax.patches):
    bar.set_alpha(df_plot['alpha'].iloc[i])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticklabels([])
ax.set_yticks([])
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)
#plt.xlabel('Continent', fontsize=20)
plt.xlabel('')
#plt.ylabel('Sum of Values', fontsize=12)
#plt.title('Predicted values by Continent', fontsize=16)
plt.yscale('log')
plt.ylabel('')
plt.xticks(fontsize=16)  # Set x-axis tick font size
plt.yticks(fontsize=16)
# ax.spines['left'].set_visible(False)
# ax.set_yticklabels([])
# ax.set_yticks([])
plt.xticks(rotation=45, ha='right')
ax.spines['left'].set_visible(False)
ax.set_yticklabels([])
ax.set_yticks([])
plt.savefig('Images/sub3.png', bbox_inches='tight')
plt.show()

# =============================================================================
# Risk Countries
# =============================================================================

pred_risk = world.sort_values('value',ascending=False)[['name','value','hist']][:10]
df_plot =pred_risk.set_index('name').sort_values('value',ascending=True)
df_plot['color'] = np.where(df_plot['value'] > df_plot['hist'], 'red', 'black')
def calculate_alpha(row):
    diff_ratio = abs(row['value'] - row['hist']) / (row['hist']+1)
    return np.clip(diff_ratio / 2 +0.5 , 0, 1)
df_plot['alpha'] = df_plot.apply(calculate_alpha, axis=1)
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=df_plot.index, y='value', data=df_plot, palette=df_plot['color'])
for i, bar in enumerate(ax.patches):
    bar.set_alpha(df_plot['alpha'].iloc[i])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)
#plt.xlabel('Country', fontsize=20)
plt.xlabel('')
plt.ylabel('')
#plt.title('Most risky countries (log scale)', fontsize=16)
plt.yscale('log')
plt.xticks(fontsize=16)  # Set x-axis tick font size
plt.yticks(fontsize=16)
ax.spines['left'].set_visible(False)
ax.set_yticklabels([])
ax.set_yticks([])
plt.savefig('Images/sub2.png', bbox_inches='tight')
plt.show()

# =============================================================================
# Increase risk
# =============================================================================

pred_risk = world.sort_values('value',ascending=False)[['name','value','hist']]
df_plot =pred_risk.set_index('name').sort_values('value',ascending=True)
df_plot['color'] = np.where(df_plot['value'] > df_plot['hist'], 'red', 'black')
def calculate_alpha(row):
    diff_ratio = abs(row['value'] - row['hist']) / (row['hist']+1)
    return np.clip(diff_ratio / 2 +0.5 , 0, 1)
df_plot['alpha'] = df_plot.apply(calculate_alpha, axis=1)
df_plot['diff'] = df_plot['value'] - df_plot['hist']
df_plot = df_plot.sort_values('diff')

df_plot_d = df_plot.iloc[:10]
#df_plot_d['alpha'] = 1-df_plot_d['alpha']
df_plot_d['diff'] = -df_plot_d['diff']
df_plot_d = df_plot_d.sort_values('diff',ascending=True)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=df_plot_d.index, y='diff', data=df_plot_d, palette=df_plot_d['color'])
for i, bar in enumerate(ax.patches):
    bar.set_alpha(df_plot_d['alpha'].iloc[i])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)
#plt.xlabel('Country', fontsize=20)
plt.xlabel('')
plt.ylabel('')
#plt.title('Most risky countries (log scale)', fontsize=16)
plt.yscale('log')
plt.xticks(fontsize=16)  # Set x-axis tick font size
plt.yticks(fontsize=16)
ax.spines['left'].set_visible(False)
ax.set_yticklabels([])
ax.set_yticks([])
plt.savefig('Images/sub2_d.png', bbox_inches='tight')
plt.show()

df_plot_d = df_plot.iloc[-10:]

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=df_plot_d.index, y='diff', data=df_plot_d, palette=df_plot_d['color'])
for i, bar in enumerate(ax.patches):
    bar.set_alpha(df_plot_d['alpha'].iloc[i])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)
#plt.xlabel('Country', fontsize=20)
plt.xlabel('')
plt.ylabel('')
#plt.title('Most risky countries (log scale)', fontsize=16)
plt.yscale('log')
plt.xticks(fontsize=16)  # Set x-axis tick font size
plt.yticks(fontsize=16)
ax.spines['left'].set_visible(False)
ax.set_yticklabels([])
ax.set_yticks([])
plt.savefig('Images/sub2_i.png', bbox_inches='tight')
plt.show()


# =============================================================================
# Specific
# =============================================================================
pred_risk = world.sort_values('value',ascending=False)[['name','value','hist']]
df_plot =pred_risk.set_index('name').sort_values('value',ascending=True)


h_train=10
h=6
pred_tot=[]
i = df_tot_m.columns.tolist().index(df_plot.index[-1])
shape = Shape()
shape.set_shape(df_tot_m.iloc[-h_train:,i]) 
find = finder(df_tot_m.iloc[:-h,:],shape)
find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
min_d_d=0.05
while len(find.sequences)<3:
    min_d_d += 0.05
    find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=2)

pred_ori = find.predict(horizon=h,plot=False,mode='mean')
seq_pred =find.predict(horizon=h,plot=False,mode='mean',seq_out=True)

plt.figure(figsize=(10, 6))
plt.plot(df_tot_m.iloc[-h_train:,i], marker='o', color='black', linestyle='-', linewidth=2, markersize=8)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel('Date', fontsize=25)
plt.xticks(fontsize=20)  # Set x-axis tick font size
plt.yticks(fontsize=20)
plt.box(False)
plt.xticks(rotation=45, ha='right')
plt.savefig('Images/ex1.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(15, 9))
for i in range(len(seq_pred[:30])):
    norm = (find.sequences[i][0] - find.sequences[i][0].min()) / (find.sequences[i][0].max()-find.sequences[i][0].min())
    norm.index = range(12-len(norm),12)
    pre = pd.Series([norm.iloc[-1]]+seq_pred.iloc[i,:].tolist())
    pre.index = range(11,18)
    plt.plot(norm,alpha=0.5,color='black')
    plt.plot(pre,alpha=0.3,color='#E41B17')
pre = pd.Series([shape.values[-1]]+pred_ori.iloc[:,0].tolist())
pre.index = range(11,18)    
plt.plot(pre,marker='o',alpha=1,linewidth=5,color='r',label='Mean value of Matches')
if max(pre)>1:
    plt.ylim(0,max(pre)+0.1*max(pre))
else:
    plt.ylim(0,1)
plt.xticks([*range(18)],[f't-{i}' for i in range(1,12)][::-1]+['t']+[f't+{i}' for i in range(1,7)])
plt.xticks([*range(18)],['']*12+[f't+{i}' for i in range(1,7)])
plt.box(False)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=20)  # Set x-axis tick font size
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.savefig('Images/ex1_all.png', bbox_inches='tight')
plt.show()
    
i = df_tot_m.columns.tolist().index(df_plot.index[-1])
name_sc=['Decrease','Stable','Increase']
col = [(216/255, 134/255, 141/255),'orangered','darkred']

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

for k in range(3):
    axs[k].plot(df_next[k].iloc[:, i], color='black', linewidth=3)
    axs[k].plot(df_next[k].iloc[-7:, i], color=col[k], linewidth=8)
    axs[k].plot(df_next[k].iloc[-6:, i], color=col[k], marker='o', linewidth=0)
    axs[k].set_frame_on(False)
    axs[k].grid(axis='y', linestyle='--', alpha=0.7)
    axs[k].tick_params(axis='y', labelsize=30)
    axs[k].set_xticks([])

plt.tight_layout()
plt.savefig('Images/ex1_sce.png', bbox_inches='tight')
plt.show()
    
#     pred_ori= seq_pred.iloc[k,:]*(find.sequences[k][0].max()-find.sequences[k][0].min())+find.sequences[k][0].min()
#     pred_ori =pred_ori.T
#     pred_ori.index = pd.date_range(start=find.sequences[k][0].index[-1],periods=h+1,freq='M')[1:]
#     historical_series = pd.concat([find.sequences[k][0],pred_ori],axis=0)
#     date_rng = historical_series.index
#     plt.figure(figsize=(10, 6))
#     plt.plot(date_rng[:-h], historical_series[:-h], marker='o', color='grey', linestyle='-', linewidth=2, markersize=8)
#     #plt.plot(date_rng[-h:], historical_series[-h:], marker='o', color='red', linestyle='-', linewidth=2, markersize=8)
#     #plt.scatter(date_rng[-h:], historical_series[-h:], color='red', s=100, zorder=5)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.xlabel('Date', fontsize=20)
#     plt.xticks(fontsize=16)  # Set x-axis tick font size
#     plt.yticks(fontsize=16)
#     plt.box(False)
#     plt.xticks(rotation=45, ha='right')
#     plt.title(f"{find.sequences[k][0].name}\nd = {find.sequences[k][1]}", style='italic', color='grey',fontsize=20)
#     plt.savefig(f'Images/ex1_m{k}.png', bbox_inches='tight')
#     plt.show()    

# series_names, values = zip(*find.sequences)
# name_barh=[]
# for i in find.sequences:
#     name_barh.append(str(i[0].name)+': '+str(i[0].index[0].month)+'-'+str(i[0].index[0].year))
# values=values[:10]
# name_barh=name_barh[:10]
# fig, ax = plt.subplots()
# bars = ax.barh(name_barh, values, color='gray', edgecolor='none')
# ax.set_frame_on(False)
# ax.grid(False)
# for bar, value in zip(bars, values):
#     ax.text(value+0.1*value, bar.get_y() + bar.get_height() / 2, f'{value:.2f}',
#             va='center', ha='center', color='gray', fontsize=15)
# ax.set_xlabel('Distance',fontsize=16)
# ax.xaxis.set_visible(False)
# ax.tick_params(axis='y', which='major', labelsize=15)
# plt.savefig('Images/ex1_barh.png', bbox_inches='tight')
# plt.show()











for coun in range(2,5):

    h_train=10
    h=6
    pred_tot=[]
    i = df_tot_m.columns.tolist().index(df_plot.index[-coun])
    shape = Shape()
    shape.set_shape(df_tot_m.iloc[-h_train:,i]) 
    find = finder(df_tot_m.iloc[:-h,:],shape)
    find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
    min_d_d=0.05
    while len(find.sequences)<3:
        min_d_d += 0.05
        find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=2)
    pred_ori = find.predict(horizon=h,plot=False,mode='mean')
    #pred_ori = pred_ori*(df_tot_m.iloc[-h_train:,i].max()-df_tot_m.iloc[-h_train:,i].min())+df_tot_m.iloc[-h_train:,i].min()
    seq_pred =find.predict(horizon=h,plot=False,mode='mean',seq_out=True)
    
    plt.figure(figsize=(15, 9))
    for j in range(len(seq_pred[:30])):
        norm = (find.sequences[j][0] - find.sequences[j][0].min()) / (find.sequences[j][0].max()-find.sequences[j][0].min())
        norm.index = range(12-len(norm),12)
        pre = pd.Series([norm.iloc[-1]]+seq_pred.iloc[j,:].tolist())
        pre.index = range(11,18)
        plt.plot(norm,alpha=0.5,color='black')
        plt.plot(pre,alpha=0.3,color='#E41B17')
    pre = pd.Series([shape.values[-1]]+pred_ori.iloc[:,0].tolist())
    pre.index = range(11,18)    
    plt.plot(pre,marker='o',alpha=1,linewidth=5,color='r')
    if max(pre)>1:
        plt.ylim(0,max(pre)+0.1*max(pre))
    else:
        plt.ylim(0,1)
    plt.xticks([*range(18)],[f't-{i}' for i in range(1,12)][::-1]+['t']+[f't+{i}' for i in range(1,7)])
    plt.xticks([*range(18)],['']*12+[f't+{i}' for i in range(1,7)])
    plt.box(False)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)  # Set x-axis tick font size
    plt.yticks(fontsize=16)
    plt.savefig(f'Images/ex{coun}_all.png', bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_tot_m.iloc[-h_train:,i], marker='o', color='black', linestyle='-', linewidth=2, markersize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Date', fontsize=25)
    plt.xticks(fontsize=20)  # Set x-axis tick font size
    plt.yticks(fontsize=20)
    plt.box(False)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'Images/ex{coun}.png', bbox_inches='tight')
    plt.show()
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    for k in range(3):
        if df_next[k].iloc[:, i].isna().all() :
            axs[k].set_frame_on(False)
            axs[k].set_xticks([])
            axs[k].set_yticks([])
        else:
            axs[k].plot(df_next[k].iloc[:, i], color='black', linewidth=3)
            axs[k].plot(df_next[k].iloc[-7:, i], color=col[k], linewidth=8)
            axs[k].plot(df_next[k].iloc[-6:, i], color=col[k], marker='o', linewidth=0)
            axs[k].set_frame_on(False)
            axs[k].grid(axis='y', linestyle='--', alpha=0.7)
            axs[k].tick_params(axis='y', labelsize=30)
            axs[k].set_xticks([])
    plt.tight_layout()
    plt.savefig(f'Images/ex{coun}_sce.png', bbox_inches='tight')
    plt.show()
    
indo=[]
for coun in range(1,5):
    indo.append(df_tot_m.columns.tolist().index(df_plot.index[-coun]))
indo.reverse()
df_best = pd.DataFrame(df_plot.index[-4:])
df_best['find']=indo
df_best.to_csv('best.csv')



