# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:22:10 2023

@author: ASUS
"""
import time
from my_dlib_funcs import *

print(os.getcwd())

# Set working paths
database_path =  os.getcwd() + '/database'
test_path = os.getcwd() + '/testing imgs'
output_path = os.getcwd() +'/Outputs'

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



beg = time.time()
db_face_descriptors = get_database_face_descriptors(database_path = database_path,
                                                    detection_scheme = 'cnn',
                                                    face_detector_path = cnn_model_path, 
                                                    shape_predictor = predictor, 
                                                    face_recognizer = face_rec , 
                                                    upsampling = 1)
print(time.time() - beg)



print(len(db_face_descriptors))
print(type(db_face_descriptors))
print(type(db_face_descriptors[0]))

for desc in db_face_descriptors:
    print(desc.keys())
    print(desc["name"])
    print(desc["img path"])

print("********************************************************")
print(len(os.listdir(test_path)))


 # define a video capture object
vid = cv2.VideoCapture(0)

 # define an output VideoWriter  object
out = cv2.VideoWriter('Me_output_vid.avi', 
                         cv2.VideoWriter_fourcc(*'MP43'),
                         1, (480, 640))


while(True):
      
    # Capture the video frame
    # by frame
    beg = time.time()
    ret, frame = vid.read()
    print(frame.shape)
    #frame = cv2.resize(frame,(320,240))
    # Display the resulting frame
    #cv2.imshow('frame', frame)
        
    descriptors = get_face_descriptors(frame  ,                                            
                                       detection_scheme='HOG',
                                       shape_predictor = predictor, 
                                       face_recognizer = face_rec , 
                                       upsampling = 1)
    
    recognize(target_descriptors = descriptors,
              database_descriptors = db_face_descriptors, 
              max_dist_thresh = 0.58 )
    
    for desc in descriptors:
        print(len(descriptors))
        print(desc["name"])
        left = desc["bounding box"].left()
        top = desc["bounding box"].top()
        right = desc["bounding box"].right()
        bottom = desc["bounding box"].bottom()
        
        img_resized  = cv2.rectangle(frame ,(left,top),(right,bottom),(255,0,0),thickness = 4)
        img_resized  = cv2.putText(frame , desc["name"], (left - 5 ,top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #out.write(frame)
        
    cv2.imshow("recognising", frame )
    end = time.time()
    fps = 1/(end - beg)
    i+=1
    print(f'FPS = {fps:.2f}')
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
