# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 20:00:04 2023

@author: ASUS
"""

"""
This python script converts a database of images into a CSV file 
by extracting face descriptors and bounding boxes from each image in the directory. 
The extracted data is organized into dictionaries, which are then written to the CSV file 
in a structured format with four columns for "name", "img path" , "bounding box" and "face descriptor".
The resulting CSV file can be used for face recognition and machine learning training or prediction. 
The script provides a convenient and efficient way to access the face data.
"""

from my_dlib_funcs import *
import time 


# the directory that contains images used as reference i.e the people to be indentified/recognised
database_path =  os.getcwd() + '/database'


# Set models paths
cnn_model_path = os.getcwd() + '/models/mmod_human_face_detector.dat'
shape_predictor_path = os.getcwd() + '/models/shape_predictor_68_face_landmarks_GTX.dat'
face_recognition_model_path = os.getcwd() + "/models/dlib_face_recognition_resnet_model_v1.dat"

# Load the shape predictor and face recognition models

predictor = dlib.shape_predictor(shape_predictor_path)
face_rec = dlib.face_recognition_model_v1(face_recognition_model_path)

# Load the CNN detection model

cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)




filename = 'people_2.csv'


update_database_descriptors(database_path = database_path,
                            detection_scheme = 'cnn',
                            csv_file = filename,
                            face_detector_path = cnn_model_path, 
                            shape_predictor = predictor, 
                            face_recognizer = face_rec , 
                            upsampling = 1)

