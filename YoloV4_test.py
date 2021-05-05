import cv2

TEST_VIDEO_FILE_NAME = "video/test.mp4"
MIN_CONFIDENCE_THRESHOLD = 0.5
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_FPS = 30

SKIP_FRAMES_TO_SPEEDUP = True
SKIP_FRAME_COUNT = 6
DETECT_ALL_CLASSES = False
desiredClasses = ["person", "car", "motorbike", "bus", "truck", "sheep"]

SAVE_DETECTED_OBJECT_IMAGES = True

LOOKING_FOR = []
classNames = []
with open("model/classList.txt", "r") as f:
    for name in f.readlines():
        name = name.strip()
        classNames.append(name)
        if not DETECT_ALL_CLASSES:
            if name in desiredClasses:
                LOOKING_FOR.append(len(classNames) - 1)
vc = cv2.VideoCapture(TEST_VIDEO_FILE_NAME)

net = cv2.dnn.readNet("model/yolov4.weights", "model/yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

skip = 0

writer = 0

objectCount = 0
while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        if SAVE_OUTPUT_VIDEO:
            writer.release()
        exit()

    if SAVE_OUTPUT_VIDEO and writer == 0:
        frameHeight, frameWidth, _ = frame.shape
        writer = cv2.VideoWriter('video/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), OUTPUT_VIDEO_FPS,
                                 (frameWidth, frameHeight))

    if SKIP_FRAMES_TO_SPEEDUP:
        if skip >= SKIP_FRAME_COUNT:
            skip = 0
        else:
            skip += 1
            continue

    classes, scores, boxes = model.detect(frame, MIN_CONFIDENCE_THRESHOLD, 0.4)

    for (id, score, box) in zip(classes, scores, boxes):
        if not DETECT_ALL_CLASSES:
            if id[0] not in LOOKING_FOR:
                continue

        if SAVE_DETECTED_OBJECT_IMAGES:
            objectCount += 1
            x, y, w, h = box
            objectROI = frame[y:y+h, x:x+w]
            cv2.imwrite("objects/"+classNames[id[0]]+str(objectCount)+".jpg", objectROI)

        cv2.rectangle(frame, box, (255,0,0), 2)
        cv2.putText(frame, classNames[id[0]], (box[0], box[1] - 5), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)

    cv2.imshow("detections", frame)
    if SAVE_OUTPUT_VIDEO:
        writer.write(frame)