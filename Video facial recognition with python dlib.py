import time
from my_dlib_funcs import *

print(os.getcwd())


SCALING_FACTOR = 0.4
# Set working paths
database_path =  os.getcwd() + '/database'
test_path = os.getcwd() + '/testing vids'
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


# get the reference face descriptors info from the people.csv file

filename = 'people.csv'

beg = time.time()
db_face_descriptors = read_db_csv(filename = filename)
print(time.time() - beg)


# display some useful info for debugging

print(len(db_face_descriptors))
print(type(db_face_descriptors))
print(type(db_face_descriptors[0]))

for desc in db_face_descriptors:
    print(desc.keys())
    print(desc["name"])
    print(desc["img path"])

print("********************************************************")



test_video_file = test_path + '/test_vid_1.MP4'
 # define a video capture object
cap = cv2.VideoCapture(test_video_file)

width  = int(cap.get(3) )  # get `width` 
height = int(cap.get(4) )  # get `height` 


print("***********************************************************")

print(width,height)
print("***********************************************************")

output_file = output_path + '/video_ouput_2.avi'
 # define an output VideoWriter  object
out = cv2.VideoWriter(output_file,
                      cv2.VideoWriter_fourcc(*"MJPG"),
                      10,(width,height))

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error opening video stream or file")


# Read the video frames
while cap.isOpened():
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        print("Error reading frame")
        break

        
      
# start recording time (to calculate FPS later)
    beg = time.time()
    print(frame.shape)
    print(width,height)

    # Capture the video frame
    # by frame
    ret, frame = cap.read()
    frame_resized = cv2.resize(frame,(int(width* SCALING_FACTOR)  , int(height * SCALING_FACTOR)))
    # Display the resulting frame
    #cv2.imshow('frame', frame)
        
    descriptors = get_face_descriptors(frame_resized  ,                                            
                                       detection_scheme='HOG',
                                       shape_predictor = predictor, 
                                       face_recognizer = face_rec , 
                                       upsampling = 1)
    
    recognize(target_descriptors = descriptors,
              database_descriptors = db_face_descriptors, 
              max_dist_thresh = 0.65 )
    
    # get the details for each detected face in the frame i.e bounding boxes and name
    for desc in descriptors:
        print(len(descriptors))
        print(desc["name"])
        
        # get bounding box coordinates
        left = int(desc["bounding box"][0] / SCALING_FACTOR)
        top = int(desc["bounding box"][1]/ SCALING_FACTOR)
        right = int(desc["bounding box"][2]/ SCALING_FACTOR)
        bottom = int(desc["bounding box"][3]/ SCALING_FACTOR)
        
        # put the face label and bounding box in the final ouput frame
        frame  = cv2.rectangle(frame ,(left,top),(right,bottom),(255,0,0),thickness = 4)
        frame  = cv2.putText(frame , desc["name"], (left - 5 ,top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # display webcam stream with results        
    cv2.imshow("Output", frame )
    #cv2.imshow("Output", frame_resized )
    
    # calculate FPS
    end = time.time()
    fps = 1/(end - beg)
    out.write(frame)
    print(f'FPS = {fps:.2f}')
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
out.release()
# Destroy all the windows
cv2.destroyAllWindows()
