


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

print(np.array(db_face_descriptors[1]["face descriptor"]).shape)
print(time.time() - beg)


# ************************************************************************************************

# header = list(db_face_descriptors[0].keys())
# print(f"keys = {header}")



# #row = db_face_descriptors.values()
# #print(row)
# rows = []
# for face in db_face_descriptors:
#     rows.append( list(face.values()))
# print('*********************************************************************************************')
# #print(row)
# print(rows)
# print('--------------------')

filename = 'people.csv'

# header = list(db_face_descriptors[0].keys())

# with open(filename, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(header)
#     #rows = [db_face_descriptor.values() for db_face_descriptor in db_face_descriptors]
#     writer.writerows(rows)

rows = []
row_dict = {}


with open(filename, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["bounding box"] = tuple(map(int,tuple(row["bounding box"][1:-1].split(', '))))
        str_pts =  row["face descriptor"].split('\n')
        row["face descriptor"] = np.array([float(str_pt) for str_pt in str_pts])
        # print('***************--------------------------------------')
        # print(type(row["face descriptor"]))
        # print(type(row["bounding box"]))
        # print(row["face descriptor"].shape)
        # print('***************--------------------------------------')

        rows.append(row)

    #rows = [row for row in reader]
    #print(rows)
    
    