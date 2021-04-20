# opencv_python

Tested with Python 3.9.2 using Pycharm CE 2021.1
https://www.jetbrains.com/pycharm/download/#section=mac

Depends on Python Module:
opencv-python==4.5.1.48

Preparation:
1. Download the Trained model from https://drive.google.com/file/d/1TUiI3lLVTSzeAWnkHcuwmnqgrV1udfb2/view?usp=sharing
2. Place the downloaded file under the "models" directory

Now the project should run :)

Customization parameters on YoloV4.py:

MIN_CONFIDENCE_THRESHOLD = 0.5

SKIP_FRAMES_TO_SPEEDUP = False
SKIP_FRAME_COUNT = 60
DETECT_ALL_CLASSES = False
desiredClasses = ["person", "car", "motorbike", "bus", "truck", "sheep"]

