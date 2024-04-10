# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:38:09 2024

@author: thoma
"""

import smtplib
import ssl
import pandas as pd
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_SERVER = "smtp.gmail.com"
PORT = 587
EMAIL = 'thomas.schincariol@gmail.com'
PASSWORD = 'hfzd vsdd zrjo spnt'

subject = "Tweet of the week"
body = "Hello Thomas, This is the tweet tentative. To send it : https://github.com/ThomasSchinca/Pace-map-risk/actions/workflows/send_tweet.yml"
receiver_email = "schincat@tcd.ie"

message = MIMEMultipart()
message["From"] = EMAIL
message["To"] = receiver_email
message["Subject"] = subject

message.attach(MIMEText(body, "plain"))
count = pd.read_csv('tweet_count.csv',index_col=0)
filename = f'Images/{count.iloc[0,0]}_c.png'

with open(filename, "rb") as attachment:
    part = MIMEBase("application", "octet-stream")
    part.set_payload(attachment.read())

encoders.encode_base64(part)

part.add_header(
    "Content-Disposition",
    f"attachment; filename= {filename}",
)

message.attach(part)
text = message.as_string()

context = ssl.create_default_context()

with smtplib.SMTP(SMTP_SERVER, PORT) as server:
    server.starttls(context=context)
    server.login(EMAIL, PASSWORD)
    server.sendmail(EMAIL, receiver_email, text)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    