# Real-time facial recognition with python dlib


<p align="center">
  <img src="https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/Outputs/video_ouput%20GIF.gif" width="400" height="200">
  <img src="https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/Outputs/webcam_output%20GIF.gif" width="400" height="200">
</p>


## About the project
 
This repository contains a facial recognition system built with Python and the dlib library.
The project uses the dlib library for facial recognition and the OpenCV library for webcam and video processing. The dlib library provides a pre-trained neural network for face detection, and the OpenCV library provides tools for capturing and processing video.  
The project is divided into four main scripts:

* `my_dlib_funcs.py`: [This script](https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/my_dlib_funcs.py) contains all the functions that will be used in the project. These functions handle tasks such as detecting faces, comparing faces, and storing face information in a CSV file.

* `generate db csv.py`: [This script](https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/generate%20db%20csv.py) takes the profile images in the `database` directory and generates a CSV file (`people.csv`) containing information about each face. The information stored in the CSV file includes the face's encoding, which is a numerical representation of the face that can be used for comparison.

* `Real-time facial recognition with python dlib v2.py`: [This script](https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/Real-time%20facial%20recognition%20with%20python%20dlib%20v2.py) captures video from a webcam and uses the functions in functions.py to recognize faces in real-time. The recognized faces are then displayed on the screen with their names.

* `Video facial recognition with python dlib.py`: [This script](https://github.com/mohamedamine99/Real-time-facial-recognition-with-python-dlib/blob/main/Video%20facial%20recognition%20with%20python%20dlib.py) processes a predefined video and uses the functions in functions.py to recognize faces in the video. The recognized faces are then displayed on the screen with their names.

## Requirements

In order for this project to work properly you would need to install the following libraries python :

* `dlib` : dlib is a powerful library for computer vision and machine learning. It includes tools for facial recognition, including face detection and feature extraction. The library is well optimized and can be used for real-time facial recognition applications.
* `OpenCV`: OpenCV is an open-source computer vision library. It provides a wide range of image processing and computer vision functions, including face detection, feature extraction, and image manipulation. OpenCV can be used to capture and process live video streams, making it useful for real-time facial recognition applications.
* `numpy` : numpy is a library for scientific computing in Python. It provides tools for working with arrays, including multi-dimensional arrays, which can be useful for storing image data. For example, an image can be represented as a numpy array, which can be processed and analyzed using numpy functions. We also used it to calculate the euclidian distance between two faces.

## Code explanation



