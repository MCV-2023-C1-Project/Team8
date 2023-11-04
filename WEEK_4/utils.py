import numpy as np
import cv2
import re
from Levenshtein import distance as levenshtein_distance
import pytesseract
import matplotlib.pyplot as plt
# Path to the OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Luis\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# This function extracts a specific text pattern from a file.
def get_text_bbdd(path):

    with open(path, 'r') as f:
        line = f.readlines()
        
    # Loop through each line in the file.
    for l in line:
        # Check if the line contains a pattern that starts with "(' and ends with ')".
        if re.search(r"\('([^']+)'", l.split(',')[0]):
            # If pattern is found, return the text inside the parentheses.
            return re.search(r"\('([^']+)'", l.split(',')[0]).group(1)
        else:
            # If pattern is not found, return 'Unknown'.
            return 'Unknown'


def get_text(gray, name_bag, x, y, x_max, y_max):

    # Extract the detected text region and apply OCR using Tesseract
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(binary[y:y_max, x:x_max])
    # Clean up the extracted text
    text = re.sub(r'[0-9\n¥“«!|]', '', text)

    # Compare the extracted text to known names using the Levenshtein distance to find the closest match from the bag of names
    min_dist = 1000000
    for name in name_bag:
        dist = levenshtein_distance(text, name)
        if dist < min_dist:
            min_dist = dist
            min_word = name

    return min_word