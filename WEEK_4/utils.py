import numpy as np
import cv2, tqdm, re
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance
import pytesseract
# Path to the OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Luis\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'



# Obtain the painting image removing the background. 
# It returns the mask where 1 means painting image and 0 background.
def get_mask(gray, threshold_area=65000):
    
    # Empty mask definition
    mask = np.zeros(gray.shape, dtype=np.uint8)

    # Applying gaussian blurring and define an intelligent gradient threshold depending on 13x13 boxes
    blur = cv2.GaussianBlur(gray, (13,13), 0)
    # Threshold based on local pixel neighborhood (11x11 block size)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Two pass dilate with horizontal and vertical kernel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    dilate = cv2.dilate(thresh, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    dilate = cv2.dilate(dilate, vertical_kernel, iterations=2)

    # Find contours, filter using contour threshold area, and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Filtering the found contours by size
    counter = 0
    areas = []
    coordinates = []
    for c in cnts:
        # Shoelace formula for convex shapes
        area = cv2.contourArea(c) 
        if area > threshold_area:
            x,y,w,h = cv2.boundingRect(c) 
            areas.append((area, (x,y,w,h)))
            counter += 1

    # Sort areas and positions by area
    areas = sorted(areas, key=lambda x: x[0], reverse=True)[:3]

    # Draw bounding box on mask
    for i in range(len(areas)-1,-1,-1):
        if i > 0 and abs(areas[i][1][0] - areas[i-1][1][0]) < 190 and abs(areas[i][1][1] - areas[i-1][1][1]) < 150:
            # print('Skipping! Two masks in the same painting!')
            continue
        x,y,w,h = areas[i][1]
        coordinates.append((x,y,w,h))
        mask[y:y+h, x:x+w] = 255
    
    # Catching the 0 contours error
    if counter == 0:
        print('Error! No paintings in this image!')
        plt.imshow(gray)
        plt.show()
        plt.imshow(mask, cmap='gray')
        plt.show()

    return mask, coordinates

# Obtain the painting image removing the text. 
# It returns the mask where 1 means painting image and 0 text.
def get_mask_text(gray, name_bag):

    # Apply morphological opening and closing to enhance text-like features using a 9x9 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    #thresholding the difference to get (hopefully) only the text
    x = closing-opening
    x = (x>125).astype(np.uint8) 

    # Dilation to further enhance the text features using a 13x13 kernel
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))
    dilated = cv2.dilate(x, kernel2, iterations=2)

    # Find contours 
    ctns = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and find rectangular bounding boxes that likely represent text areas
    areas = []
    for c in ctns[0]:
        x,y,w,h = cv2.boundingRect(c)
        # Filter out rectangles based on certain geometric criteria
        n, m = gray.shape
        ratio = w/h
        relative_area = (w*h)/(n*m)
        if w > h and ratio < 12 and ratio > 1.5 and relative_area < 0.25 and x+w < m and y+h < n and h >= 30:
            # Shoelace formula for convex shapes
            areas.append((cv2.contourArea(c), (x,y,w,h)))
    
    if len(areas) == 0:
        return 0, 0, 0, 0, 'Unknown'
    areas = sorted(areas, key=lambda x: x[0], reverse=True)
    x, y, w, h = areas[0][1]

    # Merge shapes close to the main detected text region (e.g., text broken into separate regions)
    for _, shape in areas:
        if y > shape[1]-10 and y < shape[1]+10:
            if shape[0] < x:
                w = (x+w) - shape[0]
                x = shape[0]
            else:
                w = (shape[0]+shape[2]) - x


    min_word = get_text(gray, name_bag, x, y, x+w, y+h)
    # Return the bounding box of the detected text region and the closest matching name
    return [x, y, x+w, y+h, min_word]

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

# Obtain the closest k DDBB image for query images determined by the similarity function. 
# The features have been previously calculated from the developed method.
# It returns a list of lists with the k closest images for each query image. 
def compare_images(query_features, bbdd_features, k, sim_func, param=None, filter=False, combine=False, threshold_dist=1e8):
    
    result = []
    for id1,f1 in query_features.items():
        result_i = []
        for f_i in f1:
            distances = []
            for id2,f2 in bbdd_features.items():
                text_bd = get_text_bbdd(f'data/BBDD/bbdd_{str(id2).zfill(5)}.txt')

                # Use the provided similarity function for comparing both paintings
                if not filter and not combine:
                    distances.append((id2, sim_func(f_i,f2)))
                    continue
                
                for f2_i in f2:
                    # First filter those paintings that have the same author. Then, compute similarity/distance for retrieval
                    if filter and text_bd == f_i[-1][0]:
                       distances.append((id2, param*sim_func(f_i[0][0], f2_i[0][0], normalized=True) + (1-param)*sim_func(f_i[1][0], f2_i[1][0], normalized=True)))
                    
                    # Use weighted sum between the color, texture and text scores as the similarity score for the retrieval
                    elif combine:
                        # If the similarity function is a distance 
                        if sim_func in [chi_squared_distance, levenshtein_distance]:
                            distances.append(
                                (id2, 
                                param[0]*sim_func(f_i[0][0], f2_i[0][0], normalized=True) + 
                                param[1]*sim_func(f_i[1][0], f2_i[1][0], normalized=True) +
                                param[2]*(1 - custom_leveshtein_distance(f_i[2][0], text_bd, normalized=True))
                                ))
                        else:
                            distances.append(
                                (id2, 
                                param[0]*sim_func(f_i[0][0], f2_i[0][0], normalized=True) + 
                                param[1]*sim_func(f_i[1][0], f2_i[1][0], normalized=True) +
                                param[2]*custom_leveshtein_distance(f_i[2][0], text_bd, normalized=True)
                                ))
                
            #get k smallest values from distances   
            if sim_func in [chi_squared_distance, levenshtein_distance]:
                k_smallest = sorted(distances, reverse=False, key=lambda x: x[1])[:k]
            else:
                k_smallest = sorted(distances, reverse=True, key=lambda x: x[1])[:k]

            if k_smallest[0][1] <  threshold_dist and len(f1) == 1:
                result_i.append((id1, [[-1]]))
            else:
                result_i.append((id1, k_smallest))
        
        result.append(result_i)

    # Transform the result into the required format
    result2 = []
    for x in result:
        result2_i = []
        for y in x:
            result2_i.append([z[0] for z in y[1]])
        result2.append(result2_i)
    
    return result2

def compare_keypoints(features_query, features_db, k, sim_func, threshold_matches=190):    
    bf = cv2.BFMatcher(sim_func, crossCheck=False)
    result = []
    for id_q, f_query in tqdm.tqdm(features_query.items(), desc='Computing matches'):
        result_i = []
        for f in f_query:
            number_matches = []
            for id_db, f_db in features_db.items():
                
                matches = bf.knnMatch(f, f_db[0], k=2)

                good_matches = []
                for m, n in matches: #first and second match
                    if m.distance < 0.7 * n.distance: # the first match is at least 30% closer than the second one
                        good_matches.append(m)
                number_matches.append((id_db, len(good_matches)))

            number_matches = sorted(number_matches, reverse=True, key=lambda x: x[1])[:k]

            # If the number of matches is below a certain threshold, we consider that the query image is not in the database
            if number_matches[0][1] < threshold_matches and len(f_query) == 1:
                result_i.append((id_q, [[-1]]))
            else:
                result_i.append((id_q, number_matches))
                
        result.append(result_i)

    # Transform the result into the required format
    result2 = []
    for x in result:
        result2_i = []
        for y in x:
            result2_i.append([z[0] for z in y[1]])
        result2.append(result2_i)

    return result2

def calculate_f1_score(retrievals, ground_truths):
    # Initialize counts
    TP = 0
    FP = 0
    FN = 0
    
    for retrieval, truth in zip(retrievals, ground_truths):
        # If both the retrieval and truth are -1, increment TP
        if truth == [-1] and retrieval == [[-1]]:
            TP += 1
            continue
        
        
        # If the truth is not -1 but retrieval is, it's a false negative
        if truth != [-1] and retrieval == [[-1]]:
            FN += 1
            continue

        # If the truth is -1 but retrieval is not, it's a false positive
        if truth == [-1] and retrieval != [[-1]]:
            FP += 1
            continue
        
        # For other cases where truth is not -1
        for i,sublist in enumerate(retrieval):
            if truth[i] in sublist:
                TP += 1
            else:
                FP += 1
    
    # Calculate precision, recall, and F1 score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return F1

# Copied from https://github.com/benhamner/Metrics -> Metrics.Python.ml_metrics.average_precision.py
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

# Copied from https://github.com/benhamner/Metrics -> Metrics.Python.ml_metrics.average_precision.py
def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    result = []
    for a,p in zip(actual, predicted):
        for a_i, p_i in zip(a,p):
            result.append(apk([a_i],p_i,k))
    return np.mean(result)

# compute the histogram intersection between two feature vectors
def histogram_intersection(hist1, hist2, normalized=False):
    if normalized:
        return np.sum(np.minimum(hist1, hist2)) / np.sum(np.maximum(hist1, hist2))
    else:
        return np.sum(np.minimum(hist1, hist2))

# compute the chi-squared distance between two feature vectors
def chi_squared_distance(hist1, hist2):
    return np.sum(np.square(hist1 - hist2) / (hist1 + hist2 + 1e-10))

# compute the euclidean distance between two feature vectors
def euclidean_distance(hist1, hist2):
    return np.sqrt(np.sum(np.square(hist1 - hist2)))

def custom_leveshtein_distance(s1, s2, normalized=False):
    if normalized:
        return (max(len(s1), len(s2)) - levenshtein_distance(s1, s2)) / max(len(s1), len(s2))
    else:
        return levenshtein_distance(s1, s2)