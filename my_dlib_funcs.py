# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:22:54 2023

@author: ASUS
"""

import os
import dlib
import cv2
import numpy as np


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
                         num_jitters = 1,
                         upsampling = 1 ):
    """Compute face descriptors for faces in an image.

    Args:
        frame: a numpy array representing an image.
        detection_scheme: a string indicating the face detection scheme to be
            used, either 'cnn' or 'HOG'.
        face_detector_path: the path to the cnn face detection model.
        shape_predictor: a dlib shape predictor object.
        face_recognizer: a dlib face recognizer object.
        upsampling: the upsampling factor to be used in the HOG face detection
            scheme.

    Returns:
        A list of dictionaries, each containing two items:
        - 'face descriptor': a numpy array representing the 128-dimensional face
            descriptor.
        - 'bounding box': the bounding box of the face in the image.

    Raises:
        None
    """
    
    face_descriptors = []
    if detection_scheme == 'cnn':
        
        face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
        faces = face_detector(frame)
    
        print("Number of faces detected: {}".format(len(faces)))
        for i, d in enumerate(faces):   
            cache = {}
            shape = shape_predictor(frame, d.rect)
            face_descriptor = face_recognizer.compute_face_descriptor(frame, shape ,
                                                                      num_jitters =num_jitters)                       
            
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
            face_descriptor = face_recognizer.compute_face_descriptor(frame, shape,
                                                                      num_jitters = num_jitters)                       
            
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
                                  num_jitters = 10,
                                  upsampling = 1):
    
    """This function is used to obtain face descriptors for all images in a given database path.

Args:
- database_path: str, the path to the directory that contains the images.
- detection_scheme: str, the detection scheme to be used either "cnn" or "HOG".
- face_detector_path: the path to the cnn face detector, required only if the detection_scheme is "cnn".
- shape_predictor: a dlib shape predictor, required for both detection_schemes.
- face_recognizer: a dlib face recognizer.
- upsampling: int, the number of times to upsample the image prior to applying face detection, required only if the detection_scheme is "HOG".

Returns:
- db_descriptors: list, a list of dictionaries, each dictionary contain the following keys:
    - face descriptor: numpy array, the face descriptor.
    - bounding box: dlib rect, the bounding box of the face in the image.
    - name: str, the name of the person.
    - img path: str, the path to the image in the database.
    """

    db_descriptors = []
    for i,f in enumerate(os.listdir(database_path)): #☺ **************************************

        img = dlib.load_rgb_image(database_path +'/' + f)
        face_descriptors = get_face_descriptors(img,
                                                detection_scheme = detection_scheme,
                                                face_detector_path = face_detector_path, 
                                                shape_predictor = shape_predictor, 
                                                face_recognizer = face_recognizer ,
                                                num_jitters = num_jitters,
                                                upsampling = 1)
        
        face_descriptors = face_descriptors[0]
        face_descriptors['name']= f[:-4]
        face_descriptors['img path']= database_path +'/' + f

        db_descriptors.append(face_descriptors)
        
    return db_descriptors
        
# ****************************************************************************************

def recognize(target_descriptors = None, database_descriptors = None, max_dist_thresh = 0.55 ):
    
    """Recognize faces in the target descriptors.

Given the target descriptors and database descriptors, the function finds the
best match and assigns the name to the target descriptors if the distance between
them is less than the maximum distance threshold.

Args:
    target_descriptors: A list of dictionaries, each containing the face descriptor
        and bounding box of the target face.
    database_descriptors: A list of dictionaries, each containing the face descriptor
        and name of the database face.
    max_dist_thresh: A float, representing the maximum distance threshold for face
        recognition. A face is considered recognized if the distance between the
        target face descriptor and the database face descriptor is less than the
        threshold.

Returns:
    None
    """
    
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
    


