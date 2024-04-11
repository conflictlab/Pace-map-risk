# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:40:19 2023

@author: thoma
"""

import pandas as pd
import tweepy

count = pd.read_csv('tweet_count.csv',index_col=0)
tweet_text = pd.read_csv('tweet_text.csv',index_col=0) 

# =============================================================================
# Thomas Schincat
# =============================================================================

# API_key ='iKE87AGJhYpj2dlZpBba57Ohf'
# key_secret ='w3RYtCA3kQJc0wImvEYWTQu5iTY9MrA0riVBceehAVT5Zomvy6'
# access_token='1618319345720909838-axd6L7hd3q9f7ISYQ4kHrb7mH2Btw1'
# secret_token='JlaVU540DRUSPbarc70SFj1EzxTpjJ2QmmAwyFK3dhRzE'
# client_id = 'TU54MnlxRG9ySV9xajRvaDRDbWI6MTpjaQ'
# client_sec='f9BlnK1v6AHAl198Ap4DYzv8NQo38G-Vd5M8wSSvUQk6xfUNgp'
# bearer_tok='AAAAAAAAAAAAAAAAAAAAAHKJrQEAAAAA16QgTBO%2B1Fq10JsVkzWFvVgTNMw%3D6s601sGIPqThS2pQMwMhvJHDcBRNL2rlbAVNUKJvxntyAae7cv'

# =============================================================================
# Pace account
# =============================================================================

API_key ='qg3h6s4FJLF86KjaCXUSuwZUU'
key_secret ='c60Sf12njLHbkNF7yKsNqojLgGuK9jctPQEutVELhirnvouFXF'
access_token='1521454820644773888-QTqlXt9PrHqK1L28KSvAIJvkZ1xQt9'
secret_token='aIAPQlACDYl7zmvx4jfP4Zn4yyXg19p6WFjAVv9rp7vB8'
client_id = 'dTRhSy1RUFloTldpeWFfX2FvX0g6MTpjaQ'
client_sec='l54gjxDqQTfu0ZC1ZwA1ISJT6GUGtvVhs5L8tBvfs5x9b9LfuZ'
bearer_tok='AAAAAAAAAAAAAAAAAAAAAIIEtQEAAAAAzjqCdOepSJAJ6QxabIdtGx%2BCtQw%3Dif4zijUWkbIpd6etlwwUa2h6QwtCnnV8Pt80tZozt37enxTtCG'


auth = tweepy.OAuth1UserHandler(API_key, key_secret)
auth.set_access_token(access_token, secret_token)

api = tweepy.API(auth)
media = api.media_upload(f'Images/{count.iloc[0,0]}_c.png')
media_id = media.media_id

country=tweet_text.iloc[count.iloc[0,0],0]
sta=tweet_text.iloc[count.iloc[0,0],1]
message = f'''📊 Check out the last @{country}'s fatalities pattern that mostly led to {sta} conflict risk in history. 
Explore more:
🗺️ Webapp https://thomasschinca.github.io/Pace-map-risk/
📄 Monthly Report https://www.forecastlab.org/predictionmap
💻 Github repo https://github.com/ThomasSchinca/Pace-map-risk'''

client = tweepy.Client(bearer_token=bearer_tok,consumer_key=API_key,consumer_secret=key_secret,access_token=access_token,access_token_secret=secret_token)
client.create_tweet(text=message, media_ids=[media_id])
count = count.add(1)
count.to_csv('tweet_count.csv') 
