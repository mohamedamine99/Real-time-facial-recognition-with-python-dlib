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


