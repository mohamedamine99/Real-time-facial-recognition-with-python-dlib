# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:59:23 2023

@author: ASUS
"""
from my_dlib_funcs import *
import time 
import csv

# ************************************************************************************************


database_path =  os.getcwd() + '/database'


# Set models paths
cnn_model_path = os.getcwd() + '/models/mmod_human_face_detector.dat'
shape_predictor_path = os.getcwd() + '/models/shape_predictor_68_face_landmarks_GTX.dat'
face_recognition_model_path = os.getcwd() + "/models/dlib_face_recognition_resnet_model_v1.dat"

# Load the shape predictor and face recognition models

predictor = dlib.shape_predictor(shape_predictor_path)
face_rec = dlib.face_recognition_model_v1(face_recognition_model_path)

# Load the CNN and HOG face detection models

cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
HOG_face_detector = dlib.get_frontal_face_detector()


# get the reference face descriptors from the database directory

beg = time.time()
db_face_descriptors = get_database_face_descriptors(database_path = database_path,
                                                    detection_scheme = 'cnn',
                                                    face_detector_path = cnn_model_path, 
                                                    shape_predictor = predictor, 
                                                    face_recognizer = face_rec , 
                                                    upsampling = 1)
print(time.time() - beg)


# ************************************************************************************************

header = list(db_face_descriptors[0].keys())
print(f"keys = {header}")



row = db_face_descriptors[0].values()
print(row)
row = list(row)
print('*********************************************************************************************')
print(row)


filename = 'people.csv'

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    #rows = [db_face_descriptor.values() for db_face_descriptor in db_face_descriptors]
    writer.writerows(row)


# filename = 'people.csv'

# with open(filename, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(header)
#     writer.writerows(rows)