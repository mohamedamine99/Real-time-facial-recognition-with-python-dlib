# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:22:10 2023

@author: ASUS
"""
import time
from my_dlib_funcs import *

print(os.getcwd())

database_path =  os.getcwd() + '/database'
test_path = os.getcwd() + '/testing imgs'
cnn_model_path = os.getcwd() + '/models/mmod_human_face_detector.dat'
shape_predictor_path = os.getcwd() + '/models/shape_predictor_68_face_landmarks_GTX.dat'
face_recognition_model_path = os.getcwd() + "/models/dlib_face_recognition_resnet_model_v1.dat"
output_path = os.getcwd() +'/Outputs'

cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
predictor = dlib.shape_predictor(shape_predictor_path)
face_rec = dlib.face_recognition_model_v1(face_recognition_model_path)
HOG_face_detector = dlib.get_frontal_face_detector()



beg = time.time()
db_face_descriptors = get_database_face_descriptors(database_path = database_path,
                                                    detection_scheme = 'cnn',
                                                    face_detector_path = cnn_model_path, 
                                                    shape_predictor = predictor, 
                                                    face_recognizer = face_rec , jitter = 1)
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

#◘************************************************************************************

# for f in os.listdir(test_path):
#     img = dlib.load_rgb_image(test_path +'/' +f)
#     descriptors = get_face_descriptors(img , face_detector = cnn_face_detector, 
#                         shape_predictor = predictor, face_recognizer = face_rec , jitter = 1)
    
#     for desc in descriptors:
#         #print(desc.keys())
#         fname = f[:-4]
#         #print(f[:-4])

#     recognize(target_descriptors = descriptors, 
#               database_descriptors = db_face_descriptors, 
#               max_dist_thresh = 0.58 )
    

#     for desc in descriptors:
#         print(len(descriptors))
#         print(fname + "==>" + desc["name"])
#         left = desc["bounding box"].left()
#         top = desc["bounding box"].top()
#         right = desc["bounding box"].right()
#         bottom = desc["bounding box"].bottom()
#         cv_img = cv2.rectangle(img,(left,top),(right,bottom),(255,0,0),thickness = 4)
#         cv2.imshow(desc["name"], cv_img)
        
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)

#◘************************************************************************************


    
    
        




# # define a video capture object
vid = cv2.VideoCapture(0)
out = cv2.VideoWriter('Me_output_vid.avi', 
                         cv2.VideoWriter_fourcc(*'MP43'),
                         1, (480, 640))

i = 0
skip = 1
while(True):
      
    # Capture the video frame
    # by frame
    beg = time.time()
    ret, frame = vid.read()
    print(frame.shape)
    #frame = cv2.resize(frame,(320,240))
    # Display the resulting frame
    #cv2.imshow('frame', frame)
    if i % skip == 0:
        
        descriptors = get_face_descriptors(frame  ,                                            
                                           detection_scheme='HOG',
                                           shape_predictor = predictor, 
                                           face_recognizer = face_rec , 
                                           jitter = 1)
    
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
