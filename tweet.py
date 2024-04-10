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

count = pd.read_csv('tweet_count.csv',index_col=0)
    
API_key ='iKE87AGJhYpj2dlZpBba57Ohf'
key_secret ='w3RYtCA3kQJc0wImvEYWTQu5iTY9MrA0riVBceehAVT5Zomvy6'
access_token='1618319345720909838-axd6L7hd3q9f7ISYQ4kHrb7mH2Btw1'
secret_token='JlaVU540DRUSPbarc70SFj1EzxTpjJ2QmmAwyFK3dhRzE'

client_id = 'TU54MnlxRG9ySV9xajRvaDRDbWI6MTpjaQ'
client_sec='f9BlnK1v6AHAl198Ap4DYzv8NQo38G-Vd5M8wSSvUQk6xfUNgp'
bearer_tok='AAAAAAAAAAAAAAAAAAAAAHKJrQEAAAAA16QgTBO%2B1Fq10JsVkzWFvVgTNMw%3D6s601sGIPqThS2pQMwMhvJHDcBRNL2rlbAVNUKJvxntyAae7cv'

auth = tweepy.OAuth1UserHandler(API_key, key_secret)
auth.set_access_token(access_token, secret_token)

api = tweepy.API(auth)
media = api.media_upload(f'Images\{count.iloc[0,0]}_c.png')
media_id = media.media_id

c_1='Test1'
c_2='Test2'
mon=datetime.now().strftime("%B")
ye= datetime.now().strftime("%Y")
message = f"We're observing a striking resemblance in conflict fatalities patterns: Current trends in Country @{c_1} (up to this month) closely mirror those of @{c_2} starting in @{mon}-@{ye}. The red line shows the unfolding events in @{c_2}."
client = tweepy.Client(bearer_token=bearer_tok,consumer_key=API_key,consumer_secret=key_secret,access_token=access_token,access_token_secret=secret_token)
client.create_tweet(text=message, media_ids=[media_id])

count = count.add(1)
count.to_csv('tweet_count.csv') 
