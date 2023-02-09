# -*- coding: utf-8 -*-
"""
The Python module contains a suite of functions for 
working with the dlib library for face recognition.
 The functions allow for reading and writing face data to and from a CSV database,
 extracting faces from images, and performing recognition tasks.
 
"""

import os
import csv
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
    descriptor1 (list): The first face descriptor, represented as a list of numbers.
    descriptor2 (list): The second face descriptor, represented as a list of numbers.

Returns:
    float: The Euclidean distance between the two face descriptors.


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
        num_jitters: If num_jitters>1 then each face will be randomly jittered slightly num_jitters 
            times, each run through the 128D projection, and the average used as the face descriptor.  
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
    
            left = d.rect.left()
            top = d.rect.top()
            right = d.rect.right()
            bottom = d.rect.bottom()
            
            cache["bounding box"] = (left, top, right, bottom)
            
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
            
            left = d.left()
            top = d.top()
            right = d.right()
            bottom = d.bottom()
            
            cache["bounding box"] = (left, top, right, bottom)
            
            
            
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
- num_jitters: If num_jitters>1 then each face will be randomly jittered slightly num_jitters 
    times, each run through the 128D projection, and the average used as the face descriptor.  
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
    

def save_db_to_csv(filename = '', db_face_descriptors = None):

    """Saves the given database of face descriptors to a CSV file.

    Args:
        filename: A string that specifies the name of the output CSV file. If no
            filename is provided, the default value of an empty string will be used.
        db_face_descriptors: A list of dictionaries that represent the face
            descriptors in the database. Each dictionary should have the same keys.

    Returns:
        None
    """
    
    header = list(db_face_descriptors[0].keys())

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        rows = [db_face_descriptor.values() for db_face_descriptor in db_face_descriptors]
        writer.writerows(rows)
        
# ******************************************************************************************



def read_db_csv(filename = ''):
    """Reads a CSV file that represents a database of face descriptors.

    Args:
        filename: A string that specifies the name of the input CSV file. If no
            filename is provided, the default value of an empty string will be used.

    Returns:
        A list of dictionaries that represent the face descriptors in the database.
        Each dictionary will have the following keys: "bounding box", "face descriptor".
        "bounding box" will be a tuple of integers, and "face descriptor" will be a numpy array of floats.
    """
    rows = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["bounding box"] = tuple(map(int,tuple(row["bounding box"][1:-1].split(', '))))
            str_pts =  row["face descriptor"].split('\n')
            row["face descriptor"] = np.array([float(str_pt) for str_pt in str_pts])

    
            rows.append(row)
    
    return rows

# **********************************************************************************************

def update_database_descriptors(database_path = '', 
                                csv_file = '',
                                detection_scheme = 'cnn',
                                face_detector_path = '', 
                                shape_predictor = None, 
                                face_recognizer = None ,
                                num_jitters = 10,
                                upsampling = 1 ):
    """
Updates the descriptor information of the faces in the database.

Args:
    database_path: Path to the folder that contains the images of the people 
    to be added to the database.
    
    csv_file: Path to the csv file that contains the descriptor 
    information of the people in the database.
    
    detection_scheme: The method to use for face detection.
    
    face_detector_path: Path to the face detector model file.
    
    shape_predictor: A dlib shape predictor object.
    
    face_recognizer: A dlib face recognition model object.
    
    num_jitters: Number of times to perform face alignment.
    
    upsampling: Number of times to upsample the image.

Returns:
    None

"""
    
    db_descriptors = []
    rows = read_db_csv(filename = csv_file)
    print(len(rows))
    csv_names = [row["name"] for row in rows]
    img_names = [img_name[:-4] for img_name in os.listdir(database_path)]
    
    csv_names_paths = [row["img path"] for row in rows]
    img_names_paths = {img_name[:-4] : img_name for img_name in os.listdir(database_path)}
    
    print(csv_names)
    print(img_names)
    
    if len(img_names) > len(csv_names):
        
        people_to_add = set(img_names) - set(csv_names)
        print(f"Adding {len(people_to_add)} new people to {csv_file}")
        
        for img_name in people_to_add:
            img_path = img_names_paths[img_name]
            img = dlib.load_rgb_image(database_path +'/' + img_path)
            

            face_descriptors = get_face_descriptors(img,
                                                    detection_scheme = detection_scheme,
                                                    face_detector_path = face_detector_path, 
                                                    shape_predictor = shape_predictor, 
                                                    face_recognizer = face_recognizer ,
                                                    num_jitters = num_jitters,
                                                    upsampling = 1)
            
            face_descriptors = face_descriptors[0]
            face_descriptors['name']= img_name
            face_descriptors['img path']= database_path +'/' + img_path

            db_descriptors.append(face_descriptors)
            
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            rows = [db_face_descriptor.values() for db_face_descriptor in db_descriptors]
            writer.writerows(rows)
                

                
            
        