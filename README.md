# Computer Vision
The repository consists of code for advanced lane detection, object identification and tracking. Python code trying to drive GTA San Andreas game. The game should be run in a window of size 640 x 480 from top left corner. The python code grabs the screen from top left corner and converts into images which are then used to identify the lanes and objects and tracking the objects.

## Lane detection:
Lane detection algorithm is a stripped version from Udacity's advanced lane detection using perspective view, colour coordinate transformation, Canny edge detection and moving window lane detection. The code in converted into a class and imported to the main code. OpenCV library is used.

## Object detection:
Object detection is done using Tensorflow's model API and cocoapi. The code draws boxes around the objects which have high confidence score and names them accordingly.

## Object tracking:
Object tracking is done using openCV's tracking algorithms, esp Kernelized Correlation Filters which is shown to have robust features compared to other tracking algorithms.

The codes in the repository needs tensorflow models API and cocoapi for object detection.

The tensorflow API is given in the following link.
https://github.com/tensorflow/models

Cocoapi os given in the following link.
https://github.com/cocodataset/cocoapi

![GIF](https://github.com/pradeeshvishnu/Computer-Vision/blob/master/video_gif.gif)
