# yoloV4 on video using Opencv and Python

Tested with Python 3.9.2 using Pycharm CE 2021.1
https://www.jetbrains.com/pycharm/download/#section=mac

Depends on Python Module:
opencv-python==4.5.1.48

Preparation:
1. Download the Trained model from https://drive.google.com/file/d/1TUiI3lLVTSzeAWnkHcuwmnqgrV1udfb2/view?usp=sharing
2. Place the downloaded file under the "models" directory
3. TEST_VIDEO_FILE_NAME = "video/test.mp4" modify this line on YoloV4.py if you would like to test on different sample file

Now the project should run :)

Customization parameters on YoloV4.py:
1. MIN_CONFIDENCE_THRESHOLD = 0.5
2. SKIP_FRAMES_TO_SPEEDUP = False , True SKIP_FRAME_COUNT = 60
3. DETECT_ALL_CLASSES = False, True desiredClasses = ["person", "car", "motorbike", "bus", "truck", "sheep"]









