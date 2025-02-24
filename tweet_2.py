# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:01:55 2024

@author: thoma
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.dates as mdates
import pandas as pd
import pickle 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager

font_path = 'Poppins/Poppins-Regular.ttf'
prop = FontProperties(fname=font_path)
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

def format_ticks(x, pos):
    if x == 0:
        return ''
    else:
        return '{:d}'.format(int(x))

def plot_horizontal_bar(df,names,ax,typ,maxi):
    if typ=='reg':
        li=['Africa','Americas','Asia','Europe','Middle East']
    elif typ=='dec':
        li=['90-2000','2000-2010','2010-2020','2020-Now']
    else:
        li=['<10','10-100','100-1000','>1000']
    for i,name in enumerate(li):
        if name in df.index:
            if df[name]>0:
                rect = FancyBboxPatch((0, i-0.25),width=df[name],height=0.5, boxstyle="round,pad=-0.0040,rounding_size=0.03", ec="none", fc='#808080', mutation_aspect=4)
                ax.add_patch(rect)
            else:
                pass
        else:
            pass
        
    if maxi>5:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    elif (maxi>2)&(maxi<=5):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    ax.tick_params(axis='x', labelsize=20)
    formatter = FuncFormatter(format_ticks)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_yticks([*range(len(li))],li,fontsize=20,color='#808080')
    ax.set_frame_on('y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)  
    ax.set_ylim(-0.5,len(li)-0.5)
    ax.set_xlim(0,maxi)
    ax.set_title(names,fontsize=30,pad=20,color='#808080')  
    ax.grid(axis='x', linestyle='--', alpha=0.7,color='#808080')
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_color('gray')
    

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
hist_df.index = hist_df.index.strftime('%b %y')
column_means = hist_df.mean()
threshold = 0.8
df_filtered = hist_df.loc[:, (hist_df!=0).mean() > threshold]
df_filtered = df_filtered.loc[:,df_filtered.max()>25]
with open('sce_dictionary.pkl', 'rb') as f:
    dict_sce = pickle.load(f)
with open('dict_sce.pkl', 'rb') as f:
    dict_sce_2 = pickle.load(f)
min_d_list=[]
for i in df_filtered.columns:
    min_d_list.append(len(dict_sce[i][1]))    
min_d_list = pd.Series(min_d_list,index=df_filtered.columns)
min_d_list = min_d_list.sort_values(ascending=False)

df_perc = pd.read_csv('perc.csv',index_col=(0),parse_dates=True)
df_dec = pd.read_csv('dec.csv',index_col=(0),parse_dates=True)
df_sta = pd.read_csv('sta.csv',index_col=(0),parse_dates=True)
df_inc = pd.read_csv('inc.csv',index_col=(0),parse_dates=True)
df_next=[df_dec,df_sta,df_inc]
colu_l=['grey','#F08080','#df2226']
nexto = ['Decrease','Stable','Increase']

l_coun=[]
for i in range(10):
    logo_path = 'Images/PaCE_final.png'  
    logo_img = plt.imread(logo_path)
    imagebox = OffsetImage(logo_img, zoom=0.4)  
    ab = AnnotationBbox(imagebox, (0.7, 0.75), frameon=False, xycoords='axes fraction', boxcoords="axes fraction")
    sub_name= min_d_list.index[i]
    if sub_name in rena:
        sub_name = rena[sub_name]
        
    fig, ax = plt.subplot_mosaic([[13,8,8,8,8,8,8,9,9,9,9],
                                  [13,0,0,0,0,0,0,10,10,10,10],
                                  [13,0,0,0,0,0,0,5,5,5,5],
                                  [13,0,0,0,0,0,0,5,5,5,5],
                                  [13,11,11,11,11,11,11,6,6,6,6],
                                  [13,1,1,1,2,2,2,6,6,6,6],
                                  [13,1,1,1,2,2,2,7,7,7,7],
                                  [13,3,3,3,4,4,4,7,7,7,7],
                                  [13,3,3,3,4,4,4,12,12,12,12]], figsize=(25, 18))
    ax[9].add_artist(ab)
    ax[9].text(0.05, 0.7, '@LabConflict', fontsize=35, color="lightgrey", weight='bold')
    ax[10].text(0.33, 0.5, 'What happened', fontsize=40, color="gray", ha='center', weight='bold')
    ax[10].text(0.73, 0.5, 'After ?', fontsize=40, color="#df2226", ha='center', weight='bold')
    ax[9].axis('off')
    ax[10].axis('off')
    ax[12].axis('off')
    ax[13].axis('off')
    
        
    ax[0].plot(hist_df.loc[:,min_d_list.index[i]], marker='o', color='black', linestyle='-', linewidth=2, markersize=8)
    ax[0].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    ax[0].tick_params(axis='x', labelsize=25)
    ax[0].tick_params(axis='y', labelsize=25)
    ax[0].set_title(hist_df.loc[:,min_d_list.index[i]].name, fontsize=40, font='Poppins')
    ax[0].set_frame_on(False)

    for c in range(4):
        try:
            ax[1+c].plot(dict_m[sub_name][c][0], marker='o', color='#808080', linestyle='-', linewidth=2, markersize=8)
            ax[1+c].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %y'))
            ax[1+c].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=4))
            ax[1+c].grid(axis='y', linestyle='--', alpha=0.7)
            ax[1+c].tick_params(axis='x', labelsize=18,color='#808080')
            ax[1+c].tick_params(axis='y', labelsize=18,color='#808080')
            ax[1+c].set_title(dict_m[sub_name][c][0].name, fontsize=30, font='Poppins',color='#808080')
            ax[1+c].set_frame_on(False)
        except:
            ax[1+c].axis('off')
    
    for c in range(3):
        scen = df_next[c].loc[:,sub_name]
        ax[5+c].plot(scen, color='gray', linestyle='-', linewidth=2)
        ax[5+c].plot(scen.iloc[-7:], color=colu_l[c], linestyle='-', linewidth=5)
        ax[5+c].set_frame_on(False)
        ax[5+c].set_xticks([])
        ax[5+c].set_yticks([])
        ax[5+c].spines['top'].set_visible(False)
        ax[5+c].spines['right'].set_visible(False)  
        ax[5+c].spines['bottom'].set_visible(False)
        ax[5+c].spines['left'].set_visible(False)
        ax[5+c].set_title(f'{nexto[c]} - {df_perc.loc[c,sub_name]}%',fontsize=30,color=colu_l[c])

    ax[11].text(0.5, 0.25, 'Closest historical matches', fontsize=40, color="#808080", ha='center', weight='bold')
    ax[11].axis('off')
    ax[8].text(0, 0.3, 'Last Fatalities Values', fontsize=50, color="black", ha='left', weight='bold')
    ax[8].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'Images/{i}_c.png', bbox_inches='tight')
    plt.show()
    
    if (df_perc.loc[0,sub_name]==df_perc.loc[1,sub_name]):
        if (df_perc.loc[2,sub_name]==df_perc.loc[1,sub_name]):   
            l_coun.append([hist_df.loc[:,min_d_list.index[i]].name,'a mixed'])
        else:
            l_coun.append([hist_df.loc[:,min_d_list.index[i]].name,'a slightly decreasing'])
    elif (df_perc.loc[2,sub_name]==df_perc.loc[1,sub_name]):
        l_coun.append([hist_df.loc[:,min_d_list.index[i]].name,'a slightly increasing'])
    elif (df_perc.loc[0,sub_name]>df_perc.loc[1:,sub_name].max()):
        l_coun.append([hist_df.loc[:,min_d_list.index[i]].name,'a decreasing'])
    elif (df_perc.loc[2,sub_name]>df_perc.loc[:1,sub_name].max()):
        l_coun.append([hist_df.loc[:,min_d_list.index[i]].name,'an increasing']) 
    elif (df_perc.loc[1,sub_name]>df_perc.loc[[0,2],sub_name].max()):
        l_coun.append([hist_df.loc[:,min_d_list.index[i]].name,'a stable']) 
    else:
        l_coun.append([hist_df.loc[:,min_d_list.index[i]].name,'a mixed'])
    
l_coun=pd.DataFrame(l_coun)
l_coun.to_csv('tweet_text.csv')
count=pd.Series([-1])
count.to_csv('tweet_count.csv')   



