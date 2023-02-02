# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:22:54 2023

@author: ASUS
"""

import os
import dlib
import cv2
import numpy as np
import time


dlib.DLIB_USE_CUDA = False



# ****************************************************************************************


def calculate_distance(descriptor1 = None ,descriptor2 = None):
    """
Calculates the Euclidean distance between two face descriptors.
The computed distance represents the degree of similarity between two faces.

Args:
    descriptor1 (list): The first descriptor, represented as a list of numbers.
    descriptor2 (list): The second descriptor, represented as a list of numbers.

Returns:
    float: The Euclidean distance between the two descriptors.


    """
    return np.linalg.norm(np.array(descriptor1) - np.array(descriptor2))

# ****************************************************************************************

def get_face_descriptors(frame= None ,
                         detection_scheme = 'cnn', 
                         face_detector_path = None, 
                         shape_predictor = None, 
                         face_recognizer = None , 
                         upsampling = 1 ):

    face_descriptors = []
    if detection_scheme == 'cnn':
        
        face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
        faces = face_detector(frame)
    
        print("Number of faces detected: {}".format(len(faces)))
        for i, d in enumerate(faces):   
            cache = {}
            shape = shape_predictor(frame, d.rect)
            face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)                       
            
            cache["face descriptor"] = face_descriptor
            cache["bounding box"] = d.rect
    
            
            
            face_descriptors.append(cache)
        
        return face_descriptors
    
    if  detection_scheme == 'HOG':
        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(frame,upsampling)
    
        print("Number of faces detected: {}".format(len(faces)))
        for i, d in enumerate(faces):   
            cache = {}
            shape = shape_predictor(frame, d)
            face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)                       
            
            cache["face descriptor"] = face_descriptor
            cache["bounding box"] = d
    
            
            
            face_descriptors.append(cache)
        
        return face_descriptors
        

# ****************************************************************************************

def get_database_face_descriptors(database_path = '',
                                  detection_scheme = 'cnn',
                                  face_detector_path = None, 
                                  shape_predictor = None, 
                                  face_recognizer = None , 
                                  upsampling = 1):
    db_descriptors = []
    for i,f in enumerate(os.listdir(database_path)):

        img = dlib.load_rgb_image(database_path +'/' + f)
        face_descriptors = get_face_descriptors(img,
                                                detection_scheme = detection_scheme,
                                                face_detector_path = face_detector_path, 
                                                shape_predictor = shape_predictor, 
                                                face_recognizer = face_recognizer ,
                                                upsampling = 1)
        
        face_descriptors = face_descriptors[0]
        face_descriptors['name']= f[:-4]
        face_descriptors['img path']= database_path +'/' + f

        db_descriptors.append(face_descriptors)
        
    return db_descriptors
        
# ****************************************************************************************

def recognize(target_descriptors = None, database_descriptors = None, max_dist_thresh = 0.55 ):
    
    for target_descriptor in target_descriptors:
        distances = []
        target_descriptor["name"] = 'UNKNOWN'

        for database_descriptor in database_descriptors:
            dist = calculate_distance(np.array(target_descriptor["face descriptor"]),
                                      np.array(database_descriptor["face descriptor"]))
            
            distances.append(dist)
        
        if distances:
            idx = np.argmin(distances)
            print(distances[idx])
            if distances[idx] < max_dist_thresh:
                target_descriptor["name"] = database_descriptors[idx]["name"]
        
# ****************************************************************************************
    


