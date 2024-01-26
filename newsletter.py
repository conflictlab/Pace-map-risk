# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 22:07:57 2024

@author: thoma
"""
import os
from pdf2image import convert_from_path
import base64
from io import BytesIO


def convert_pdf_to_html_base64(pdf_path, output_folder):
    # Convert PDF to a list of images
    images = convert_from_path(pdf_path)

    # HTML parts
    html_images = []

    for i, image in enumerate(images):
        # Convert image to BytesIO object
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        
        # Encode image to Base64
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Embed Base64 image in HTML
        html_images.append(f'<img src="data:image/png;base64,{img_base64}" alt="Page {i+1}" style="width:100%;">')

    # Combine all HTML parts with CSS for responsive width
    html_content = "<html><head><style>img{max-width:100%;height:auto;}</style></head><body>" + "".join(html_images) + "</body></html>"

    # Save HTML file
    html_file_path = os.path.join(output_folder, "output.html")
    with open(html_file_path, 'w') as html_file:
        html_file.write(html_content)

    return html_file_path


# Example usage
pdf_path = 'assets/Report.pdf' # Example PDF path
output_folder = 'assets/html_output/' # Output folder for HTML and images
html_file_path = convert_pdf_to_html_base64(pdf_path, output_folder)

