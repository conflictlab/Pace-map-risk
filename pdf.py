# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:27:12 2023

@author: thoma
"""

# Since the image has been uploaded, we can now proceed to add it to the PDF.
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from datetime import datetime,date
from dateutil.relativedelta import relativedelta
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import  Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd


# Path to your Poppins font .ttf files
poppins_bold_path = 'Poppins/Poppins-Bold.ttf'  # Replace with your font file path
poppins_regular_path = 'Poppins/Poppins-Regular.ttf'  # Replace with your font file path

# Register the Poppins font
pdfmetrics.registerFont(TTFont('Poppins-Bold', poppins_bold_path))
pdfmetrics.registerFont(TTFont('Poppins', poppins_regular_path))

month = datetime.now().strftime("%B")
year = datetime.now().strftime("%Y")
teasing = date.today() + relativedelta(months=+1)
month_t = teasing.strftime("%B")
s_month_t = teasing.strftime("%b")
year_t = teasing.strftime("%Y")
month_s = datetime.now().strftime("%b")
six_months = date.today() + relativedelta(months=+5)
sm_m = six_months.strftime("%b")
sm_y = six_months.strftime("%Y")


# Define the path for the new PDF and the uploaded image
output_pdf_path = 'assets/Report.pdf'
c = canvas.Canvas(output_pdf_path, pagesize=letter)
c.setTitle(f"Report {s_month_t}-{year_t}")
title_text = f'Patterns of Conflict ({month_t} {year_t} Newsletter)'
title_font = "Poppins-Bold"  
title_font_size = 22
title_color = HexColor("#df2226")
# Add title to the PDF
c.setFont(title_font, title_font_size)
c.setFillColor(title_color)
c.drawCentredString(297.5, 750, title_text)  

def draw_paragraph(c, text, font, font_size, color, x, y, width):
    styles = getSampleStyleSheet()
    style = styles["BodyText"]
    style.fontName = font
    style.fontSize = font_size
    style.textColor = color

    # Create a Paragraph object
    paragraph = Paragraph(text, style)

    # Draw the Paragraph on the canvas
    paragraph.wrapOn(c, width, 1000)  # Adjust the height as needed
    paragraph.drawOn(c, x, y)
    
subtitle_text = 'Our Global Risk Prediction Map identifies countries with similar past experiences in conflict-related fatalities. By analyzing historical data patterns, this approach forecasts future trends and highlights nations with comparable conflict trajectories.'
subtitle_font = "Poppins"  # Using a standard sans-serif font
subtitle_font_size = 11
subtitle_color = HexColor("#505050")
paragraph_x = 25
paragraph_y = 700
paragraph_width = 550  # Adjust as needed
# Draw the paragraph
draw_paragraph(c, subtitle_text, subtitle_font, subtitle_font_size, subtitle_color, paragraph_x, paragraph_y, paragraph_width)


# Set the subtitle properties
subtitle_text = "Global Risk Prediction Map ("+str(month_s)+' '+str(year)+' - '+str(sm_m)+' '+str(sm_y)+")"
subtitle_font = "Poppins"  # Using a standard sans-serif font
subtitle_font_size = 11
  # Assuming "gris foncé 3" is a dark grey color
c.setFont(subtitle_font, subtitle_font_size)
c.setFillColor(subtitle_color)
c.drawCentredString(297.5, 670, subtitle_text)  # Positioning subtitle below the title

# Place the image on the PDF
map_image_path = 'Images/map.png'
c.drawImage(map_image_path, x=70, y=440, width=450, height=225, mask='auto')  # Scaled to fit


t_2 = "Global expected Fatalities"
c.setFillColor(subtitle_color)
c.drawCentredString(297.5, 415, t_2)
sub_image1 = 'Images/sub1_1.png'
c.drawImage(sub_image1, x=25, y=250, width=550,height=160,  mask='auto')  # Scaled to fit






sub_image2 = 'Images/sub2.png'
sub_image3 = 'Images/sub2_i.png'
sub_image4 = 'Images/sub2_d.png'
c.drawImage(sub_image2, x=30, y=70, width=180,height=150, mask='auto')
c.drawImage(sub_image3, x=215, y=70, width=180,height=150, mask='auto')  
c.drawImage(sub_image4, x=410, y=70, width=180,height=150, mask='auto')  
subtitle_font_size = 10
  # Assuming "gris foncé 3" is a dark grey color
c.setFont(subtitle_font, subtitle_font_size)
t_1 = "Largest nb. of fatalities expected"
c.setFillColor(subtitle_color)
c.drawCentredString(120, 225, t_1)
t_3 = "Largest expected increase"
c.setFillColor(subtitle_color)
c.drawCentredString(310, 225, t_3)
t_4 = "Largest expected decrease"
c.setFillColor(subtitle_color)
c.drawCentredString(510, 225, t_4)


pace_logo = 'Images/PaCE_final.png'
c.drawImage(pace_logo, x=50, y=20, width=100,height=40,  mask='auto')  # Scaled to fit

info_text = "Click here for more info"
info_url = "https://shapefinderlive.azurewebsites.net/"
c.setFillColor(HexColor("#999999"))  # Blue color for the link text
c.setFont("Poppins", 10)
text_width = c.stringWidth(info_text, "Poppins", 10)
c.drawString((letter[0] - text_width) / 2, 30, info_text)
c.linkURL(info_url, (297.5 - text_width / 2, 30, 297.5 + text_width / 2, 42), relative=1)

# Adding "Twitter" and "Website" links in the bottom right corner
twitter_text = "Twitter"
twitter_url = "https://twitter.com/LabConflict"
website_text = "Website"
website_url = "https://paceconflictlab.wixsite.com/conflict-research-la"
contact = "Contact"


# Calculate the positioning for the text
right_margin = 50
bottom_margin = 20
line_height = 12



c.drawString(letter[0] - right_margin - c.stringWidth('schincat@tcd.ie', "Poppins", 10), bottom_margin + 2*line_height, 'schincat@tcd.ie')
c.drawString(letter[0] - right_margin - c.stringWidth(twitter_text, "Poppins", 10), bottom_margin + line_height, twitter_text)
c.linkURL(twitter_url, (letter[0] - right_margin - c.stringWidth(twitter_text, "Poppins", 10), bottom_margin + line_height, letter[0] - right_margin, bottom_margin + line_height * 2), relative=1)
c.drawString(letter[0] - right_margin - c.stringWidth(website_text, "Poppins", 10), bottom_margin, website_text)
c.linkURL(website_url, (letter[0] - right_margin - c.stringWidth(website_text, "Poppins", 10), bottom_margin, letter[0] - right_margin, bottom_margin + line_height), relative=1)

subtitle_font = "Poppins"  # Using a standard sans-serif font
subtitle_font_size = 11
subtitle_color = HexColor("#505050")  # Assuming "gris foncé 3" is a dark grey color
c.setFont(subtitle_font, subtitle_font_size)
c.setFillColor(subtitle_color)
c.drawString(letter[0] - right_margin - c.stringWidth('Contact', "Poppins", 11), bottom_margin + 3.25*line_height, 'Contact')
c.showPage()

df_best = pd.read_csv('best.csv',index_col=0)
df_perc = pd.read_csv('perc.csv',parse_dates=True,index_col=(0))

first = df_best.iloc[-1][0]
perc = df_perc.iloc[:,df_best.iloc[-1][1]]
title_text = f'{first}'
title_font = "Poppins-Bold"  
title_font_size = 13
title_color = HexColor("#df2226")
c.setFont(title_font, title_font_size)
c.setFillColor(title_color)
c.drawString(30, 760, title_text)


sub_image2 = 'Images/ex1.png'
sub_image3 = 'Images/ex1_all.png'
sub_image4 = 'Images/ex1_sce.png'
c.drawImage(sub_image2, x=30, y=600, width=160,height=120, mask='auto')
c.drawImage(sub_image3, x=220, y=600, width=200,height=120, mask='auto') 
c.drawImage(sub_image4, x=440, y=590, width=100,height=140, mask='auto')  

subtitle_font_size = 8
c.setFont(subtitle_font, subtitle_font_size)
t_1 = "Decrease"
c.setFillColor((216/255, 134/255, 141/255))
c.drawCentredString(565, 710, t_1)
t_1 =f'pr={perc.iloc[0]}%'
c.setFillColor((216/255, 134/255, 141/255))
c.drawCentredString(565, 695, t_1)
t_3 = "Stable"
c.setFillColor('orangered')
c.drawCentredString(565, 665, t_3)
t_3 = f"pr={perc.iloc[1]}%"
c.setFillColor('orangered')
c.drawCentredString(565, 650, t_3)
t_4 = "Increase"
c.setFillColor('darkred')
c.drawCentredString(565, 620, t_4)
t_4 = f"pr={perc.iloc[2]}%"
c.setFillColor('darkred')
c.drawCentredString(565, 605, t_4)


subtitle_font_size = 10
c.setFont(subtitle_font, subtitle_font_size)
t_1 = "Fatalities over last 10 months"
c.setFillColor(subtitle_color)
c.drawCentredString(110, 740, t_1)
t_3 = "Closest historical matches"
c.setFillColor(subtitle_color)
c.drawCentredString(320, 740, t_3)
t_4 = "Scenarios"
c.setFillColor(subtitle_color)
c.drawCentredString(525, 740, t_4)


first = df_best.iloc[-2][0]
perc = df_perc.iloc[:,df_best.iloc[-2][1]]
title_text = f'{first}'
title_font = "Poppins-Bold"  
title_font_size = 13
title_color = HexColor("#df2226")
c.setFont(title_font, title_font_size)
c.setFillColor(title_color)
c.drawString(30, 580, title_text)


sub_image2 = 'Images/ex2.png'
sub_image3 = 'Images/ex2_all.png'
sub_image4 = 'Images/ex2_sce.png'
c.drawImage(sub_image2, x=30, y=420, width=160,height=120, mask='auto')
c.drawImage(sub_image3, x=220, y=420, width=200,height=120, mask='auto')
c.drawImage(sub_image4, x=440, y=410, width=100,height=140, mask='auto')  

subtitle_font_size = 8
c.setFont(subtitle_font, subtitle_font_size)
t_1 = "Decrease"
c.setFillColor((216/255, 134/255, 141/255))
c.drawCentredString(565, 530, t_1)
t_1 =f'pr={perc.iloc[0]}%'
c.setFillColor((216/255, 134/255, 141/255))
c.drawCentredString(565, 515, t_1)
t_3 = "Stable"
c.setFillColor('orangered')
c.drawCentredString(565, 485, t_3)
t_3 = f"pr={perc.iloc[1]}%"
c.setFillColor('orangered')
c.drawCentredString(565, 470, t_3)
t_4 = "Increase"
c.setFillColor('darkred')
c.drawCentredString(565, 440, t_4)
t_4 = f"pr={perc.iloc[2]}%"
c.setFillColor('darkred')
c.drawCentredString(565, 425, t_4)


subtitle_font_size = 10
c.setFont(subtitle_font, subtitle_font_size)
t_1 = "Fatalities over last 10 months"
c.setFillColor(subtitle_color)
c.drawCentredString(110, 560, t_1)
t_3 = "Closest historical matches"
c.setFillColor(subtitle_color)
c.drawCentredString(320, 560, t_3)
t_4 = "Scenarios"
c.setFillColor(subtitle_color)
c.drawCentredString(525, 560, t_4)



first = df_best.iloc[-3][0]
perc = df_perc.iloc[:,df_best.iloc[-3][1]]
title_text = f'{first}'
title_font = "Poppins-Bold"  
title_font_size = 13
title_color = HexColor("#df2226")
c.setFont(title_font, title_font_size)
c.setFillColor(title_color)
c.drawString(30, 400, title_text)


sub_image2 = 'Images/ex3.png'
sub_image3 = 'Images/ex3_all.png'
sub_image4 = 'Images/ex3_sce.png'
c.drawImage(sub_image2, x=30, y=250, width=160,height=120, mask='auto')
c.drawImage(sub_image3, x=220, y=250, width=200,height=120, mask='auto')
c.drawImage(sub_image4, x=440, y=240, width=100,height=140, mask='auto') 

subtitle_font_size = 8
c.setFont(subtitle_font, subtitle_font_size)
t_1 = "Decrease"
c.setFillColor((216/255, 134/255, 141/255))
c.drawCentredString(565, 360, t_1)
t_1 =f'pr={perc.iloc[0]}%'
c.setFillColor((216/255, 134/255, 141/255))
c.drawCentredString(565, 345, t_1)
t_3 = "Stable"
c.setFillColor('orangered')
c.drawCentredString(565, 315, t_3)
t_3 = f"pr={perc.iloc[1]}%"
c.setFillColor('orangered')
c.drawCentredString(565, 300, t_3)
t_4 = "Increase"
c.setFillColor('darkred')
c.drawCentredString(565, 270, t_4)
t_4 = f"pr={perc.iloc[2]}%"
c.setFillColor('darkred')
c.drawCentredString(565, 255, t_4)
 

subtitle_font_size = 10
c.setFont(subtitle_font, subtitle_font_size)
t_1 = "Fatalities over last 10 months"
c.setFillColor(subtitle_color)
c.drawCentredString(110, 382, t_1)
t_3 = "Closest historical matches"
c.setFillColor(subtitle_color)
c.drawCentredString(320, 382, t_3)
t_4 = "Scenarios"
c.setFillColor(subtitle_color)
c.drawCentredString(525, 382, t_4)


first = df_best.iloc[-4][0]
perc = df_perc.iloc[:,df_best.iloc[-4][1]]
title_text = f'{first}'
title_font = "Poppins-Bold"  
title_font_size = 13
title_color = HexColor("#df2226")
c.setFont(title_font, title_font_size)
c.setFillColor(title_color)
c.drawString(30, 220, title_text)


sub_image2 = 'Images/ex4.png'
sub_image3 = 'Images/ex4_all.png'
sub_image4 = 'Images/ex4_sce.png'
c.drawImage(sub_image2, x=30, y=70, width=160,height=120, mask='auto')
c.drawImage(sub_image3, x=220, y=70, width=200,height=120, mask='auto')
c.drawImage(sub_image4, x=440, y=60, width=100,height=140, mask='auto')  

subtitle_font_size = 8
c.setFont(subtitle_font, subtitle_font_size)
t_1 = "Decrease"
c.setFillColor((216/255, 134/255, 141/255))
c.drawCentredString(565, 175, t_1)
t_1 =f'pr={perc.iloc[0]}%'
c.setFillColor((216/255, 134/255, 141/255))
c.drawCentredString(565, 160, t_1)
t_3 = "Stable"
c.setFillColor('orangered')
c.drawCentredString(565, 130, t_3)
t_3 = f"pr={perc.iloc[1]}%"
c.setFillColor('orangered')
c.drawCentredString(565, 115, t_3)
t_4 = "Increase"
c.setFillColor('darkred')
c.drawCentredString(565, 85, t_4)
t_4 = f"pr={perc.iloc[2]}%"
c.setFillColor('darkred')
c.drawCentredString(565, 70, t_4)


subtitle_font_size = 10
c.setFont(subtitle_font, subtitle_font_size)
t_1 = "Fatalities over last 10 months"
c.setFillColor(subtitle_color)
c.drawCentredString(110, 200, t_1)
t_3 = "Closest historical matches"
c.setFillColor(subtitle_color)
c.drawCentredString(320, 200, t_3)
t_4 = "Scenarios"
c.setFillColor(subtitle_color)
c.drawCentredString(525, 200, t_4)






pace_logo = 'Images/PaCE_final.png'
c.drawImage(pace_logo, x=50, y=20, width=100,height=40,  mask='auto')  # Scaled to fit

info_text = "Click here for more info"
info_url = "https://shapefinder.azurewebsites.net/"
c.setFillColor(HexColor("#999999"))  # Blue color for the link text
c.setFont("Poppins", 10)
text_width = c.stringWidth(info_text, "Poppins", 10)
c.drawString((letter[0] - text_width) / 2, 30, info_text)
c.linkURL(info_url, (297.5 - text_width / 2, 30, 297.5 + text_width / 2, 42), relative=1)

# Adding "Twitter" and "Website" links in the bottom right corner
twitter_text = "Twitter"
twitter_url = "https://twitter.com/LabConflict"
website_text = "Website"
website_url = "https://paceconflictlab.wixsite.com/conflict-research-la"
contact = "Contact"


# Calculate the positioning for the text
right_margin = 50
bottom_margin = 10
line_height = 12


c.drawString(letter[0] - right_margin - c.stringWidth('schincat@tcd.ie', "Poppins", 10), bottom_margin + 2*line_height, 'schincat@tcd.ie')
c.drawString(letter[0] - right_margin - c.stringWidth(twitter_text, "Poppins", 10), bottom_margin + line_height, twitter_text)
c.linkURL(twitter_url, (letter[0] - right_margin - c.stringWidth(twitter_text, "Poppins", 10), bottom_margin + line_height, letter[0] - right_margin, bottom_margin + line_height * 2), relative=1)
c.drawString(letter[0] - right_margin - c.stringWidth(website_text, "Poppins", 10), bottom_margin, website_text)
c.linkURL(website_url, (letter[0] - right_margin - c.stringWidth(website_text, "Poppins", 10), bottom_margin, letter[0] - right_margin, bottom_margin + line_height), relative=1)

subtitle_font = "Poppins" 
subtitle_font_size = 11
subtitle_color = HexColor("#505050")  
c.setFont(subtitle_font, subtitle_font_size)
c.setFillColor(subtitle_color)
c.drawString(letter[0] - right_margin - c.stringWidth('Contact', "Poppins", 11), bottom_margin + 3.25*line_height, 'Contact')
c.showPage()


content = """
The "Patterns of Conflict" report identifies and compares conflict patterns across various
countries. This process involves aggregating historical conflict data and matching similar
patterns of conflict-related events. The methodology focuses on identifying trends and 
potential future scenarios based on historical data. The objective is to provide a predictive 
insight into how conflict patterns may evolve, aiding in better-informed strategic planning 
and decision-making.

The methodology in the "Patterns of Conflict" report is centered on a comparative analysis of 
conflict-related data across countries. It involves the following steps:

1.  Data collection. The data used in the "Patterns of Conflict" report is sourced from the 
    Uppsala Conflict Data Program (UCDP), a comprehensive database that records and codes 
    data on conflict and associated events worldwide. Specifically, the report makes use of 
    the "best" estimate variable for battle-related deaths provided by UCDP 
    (see https://ucdp.uu.se/downloads/brd/ucdp-brd-codebook.pdf)

2.  Short sequences of casualty data are compared to each other using various algorithms 
    (DTW, Euclidean distance), which allow us to identify similar shapes in the data, even 
    ones that may be out of sync temporally. A distance threshold is applied to select only 
    sequences that are close matches.

3.  The model then predicts potential increases or decreases in conflict-related fatalities 
    based on an average of past patterns.
"""


title_text = 'About'
title_font = "Poppins-Bold"  
title_font_size = 13
title_color = HexColor("#df2226")
c.setFont(title_font, title_font_size)
c.setFillColor(title_color)
c.drawCentredString(70, 730, title_text)

y_position = letter[1]-70
# Add the content to the PDF using canvas
for line in content.split('\n'):
    y_position -= 12  # Adjust the spacing as needed
    subtitle_font = "Poppins"  # Using a standard sans-serif font
    subtitle_font_size = 11
    subtitle_color = HexColor("#505050")  # Assuming "gris foncé 3" is a dark grey color
    c.setFont(subtitle_font, subtitle_font_size)
    c.setFillColor(subtitle_color)
    c.drawString(50, y_position, line)

c.save()

output_pdf_path

