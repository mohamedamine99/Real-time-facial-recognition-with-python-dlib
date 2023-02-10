# Real-time facial recognition with python dlib


<p align="center">
  <img src="https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/Outputs/video_ouput%20GIF.gif" width="400" height="275">
  <img src="https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/Outputs/webcam_output%20GIF.gif" width="400" height="275">
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>  
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#code-explanation">Code explanation</a></li>      
  </ol>
</details>

## About the project
 
This repository contains a facial recognition system built with Python and the dlib library.
The project is an improvement upon a [previous](https://github.com/mohamedamine99/Facial-recognition-with-dlib) implementation and uses the dlib library for facial recognition and the OpenCV library for webcam and video processing. The dlib library provides a pre-trained neural network for face detection, and the OpenCV library provides tools for capturing and processing video.  
The project is divided into 6 main scripts:

* `my_dlib_funcs.py`: [This script](https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/my_dlib_funcs.py) contains all the functions that will be used in the project. These functions handle tasks such as detecting faces, comparing faces, and storing face information in a CSV file.

* `generate db csv.py`: [This script](https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/generate%20db%20csv.py) converts a database of images (stored in the `databsae` folder) into a CSV file (`people.csv`) by extracting face descriptors and bounding boxes from each image in the directory. The extracted data is organized into dictionaries, which are then written to the CSV file in a structured format with two columns for "bounding box" and "face descriptor". The resulting CSV file can be used for face recognition and machine learning training or prediction. The script provides a convenient and efficient way to access the face data.

* `updating csv db sample code.py`: [This script](https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/updating%20csv%20db%20sample%20code.py) converts the new images ,that are not already processed, of database of images (stored in the `databsae` folder) into a CSV file (`people_2.csv` for example) with the same method as above.

* `Real-time facial recognition with python dlib v2.py`: [This script](https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/Real-time%20facial%20recognition%20with%20python%20dlib%20v2.py) captures video from a webcam and uses the functions in functions.py to recognize faces in real-time. The recognized faces are then displayed on the screen with their names.

* `Video facial recognition with python dlib.py`: [This script](https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/Video%20facial%20recognition%20with%20python%20dlib.py) processes a predefined video and uses the functions in my_dlib_funcs.py to recognize faces in the video. The recognized faces are then displayed on the screen with their names.

* `Real-time tracking and facial recognition with dlib.py`: [This script](https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/Real-time%20tracking%20and%20facial%20recognition%20with%20dlib.py) processes a predefined video and uses the functions in my_dlib_funcs.py to recognize faces in the webcam stream. Unlike previous scripts, it does not perform full recognition on every single frame, as that can be computationally taxing. Instead, it alternates between detection and recognition for a set number of frames, and tracking for the remaining frames, repeatedly.

## Requirements

In order for this project to work properly you would need to install the following libraries python :

* `dlib` : dlib is a powerful library for computer vision and machine learning. It includes tools for facial recognition, including face detection and feature extraction. The library is well optimized and can be used for real-time facial recognition applications.
* `OpenCV`: OpenCV is an open-source computer vision library. It provides a wide range of image processing and computer vision functions, including face detection, feature extraction, and image manipulation. OpenCV can be used to capture and process live video streams, making it useful for real-time facial recognition applications.
* `numpy` : numpy is a library for scientific computing in Python. It provides tools for  working with arrays, including multi-dimensional arrays, which can be useful for storing image data. For example, an image can be represented as a numpy array, which can be processed and analyzed using numpy functions. We also used it to calculate the euclidian distance between two faces.

## Code explanation
 ### my_dlib_funcs.py :
 
`my_dlib_funcs.py`: contains all the functions that will be used in the project. These functions are as follows :

* `save_db_to_csv`:a function that takes two arguments, `filename` and `db_face_descriptors`. The function converts the information contained in db_face_descriptors into a CSV (Comma-Separated Values) file with the name specified by filename.

* `read_db_csv` : a function that reads a CSV file representing a database of face descriptors. The function takes one argument, a string `filename` representing the name of the input CSV file.

* `recognize` : a function that performs face recognition. It takes three inputs: 

     - `target_descriptors` is a list of dictionaries, each containing the face descriptor and bounding box of the target face.      
     - `database_descriptors` is a list of dictionaries, each containing the face descriptor and name of the database face.  
     - `max_dist_thresh` is a float representing the maximum distance threshold for face recognition. A face is considered recognized if the distance between the target face descriptor and the database face descriptor is less than the threshold.

     - This function does not return any value, but it updates the ` "name" `field in the dictionaries of the target_descriptors list.

* `calculate_distance` : a function that Calculates the Euclidean distance between two face descriptors. The computed distance represents the degree of similarity between two faces. It takes two arguments : the two face descriptors we want to compare.

* `get_face_descriptors` : a function that computes face descriptors for each of the faces in a given image . It returns the list face_descriptors containing dictionaries with face descriptors and bounding boxes of faces in the image.

* `get_database_face_descriptors` :This function is used to obtain face descriptors for all images in a given database path. It is mainly used to extract face descriptors from our database.

* `update_database_descriptors` :Updates the descriptor information of the faces in the database by appending only new unseen faces data.


 ### generate db csv.py :
 
 This python script converts a database of images into a CSV file  by extracting face descriptors and bounding boxes from each image in the directory. 
 The extracted data is organized into dictionaries, which are then written to the CSV file in a structured format with four columns for `"name"`, `"img path"` , `"bounding box"` and `"face descriptor"`.
 The resulting CSV file can be used for face recognition and machine learning training or prediction. 
 The script provides a convenient and efficient way to access the face data.
 
1. First, it imports the module `my_dlib_funcs` and the time library.
2. It sets the path to the directory containing the reference images, and the paths to the models used for face detection, landmark extraction, and face recognition.
3. The code then loads the shape predictor and face recognition models using dlib. It also loads a face detection model based on the Convolutional Neural Network (CNN). The CNN model is the one used to extract features from our database since it's much more accurate (though much more computationally expensive) than it's alternative HOG 
4. The code calls the function `get_database_face_descriptors` to obtain face descriptors for all the images in the reference database. This function uses the loaded face detection and face recognition models.It then prints the shape of the face descriptor for one of the images in the reference database, and the time it took to obtain all the face descriptors (for debugging).
6. Finally, it saves the face descriptors to a CSV file using the `save_db_to_csv` function and then reads the saved data back into memory using the "read_db_csv" function (for debugging).

 ### Real-time facial recognition with python dlib v2.py:
 
The `Real-time facial recognition with python dlib v2.py` python script utilizes the dlib library for facial recognition for real-time video stream fom a webcam.
1. The script first reads in a database of faces stored in a saved CSV file. The database contains face descriptors and bounding boxes for each face detected in a set of images.

2. It sets the path to the directory containing the paths to the models used for face detection, landmark extraction, and face recognition and the paths to the output folder.

3. The code then loads the shape predictor and face recognition models using dlib. It also loads a face detection model based on the Histogram of Oriented Gradients (HOG). The HOG model is the one used to extract features from our real-time video frames since it's much faster (much less computationally expensive but less accurate) than it's alternative CNN (Convolutional Neural Networks).

4. The database is loaded using the `read_db_csv`.

5. The script then uses the opencv functionalities so that The webcam captures images in real-time, and the `get_face_descriptors` function to detect faces in each image to compare them to the faces in the database.

6. The `recognize` function compares the faces detected in the webcam stream to the faces in the database. 
The algorithm calculates the distance between the face descriptors for each pair of faces and returns the closest match. 
If a match is found, the script will display the name of the person in the database that the live face is closest to else it would display `"UNKNOWN"`.


This script provides a demonstration of how the dlib library can be used for real-time facial recognition. 
The use of a saved CSV file for storing the face database allows for a flexible and scalable solution,as the database can be easily updated or expanded as needed. 

 ### Video facial recognition with python dlib.py :
The `Video facial recognition with python dlib.py` python script utilizes the dlib library for facial recognition for video stream fom a video file.
It follows the same steps as the `Real-time facial recognition with python dlib v2.py` python script.

