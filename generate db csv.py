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

print(np.array(db_face_descriptors[1]["face descriptor"]).shape)
print(time.time() - beg)



filename = 'people.csv'

save_db_to_csv(filename = filename , db_face_descriptors = db_face_descriptors)

db_face_descriptors = read_db_csv(filename = filename)

print(db_face_descriptors[1])