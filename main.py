import cv2
import matplotlib.pyplot as plt
import subprocess
from pygame import mixer


mixer.init()
sound = mixer.Sound(r"./sounds/beep.wav")

config = r"./config/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
model = r"./config/frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(model, config)

classLabels = []
file_name = r"./coco/coco.labels"

with open(file_name, "rt") as fpt:
    classLabels = fpt.read().rstrip("\n").split()

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# img = cv2.imread(r"./tests/road.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# scale_percent = 30

# calculate the 50 percent of original dimensions
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)

# dsize
# dsize = (width, height)

# output = cv2.resize(img, dsize)
# cv2.imwrite('./cv2-resize-image-50.png',output)

# class_index, confidence, bbox = model.detect(output, confThreshold=0.6)
# print(class_index)

# font_scale = 3
# font = cv2.FONT_HERSHEY_PLAIN
# for ClassInd, conf, boxes, in zip(class_index.flatten(), confidence.flatten(), bbox):
#   cv2.rectangle(output, boxes, (255, 0, 0), 2)
#   cv2.putText(output, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

# cv2.imshow('image', output)
# cv2.waitKey(0)

######################################### video

cap = cv2.VideoCapture(r"./videos/tcc.mp4")
# cap = cv2.VideoCapture(r'./tests/VID_20220926_130325.mp4')

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
scale_percent = 30

while cap.isOpened():
    _, frame = cap.read()

    # img = cv2.imread(r"./tests/road.png")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # calculate the 50 percent of original dimensions
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    roi = cv2.resize(frame, dsize)

    # video = cv2.rotate(frame, cv2.ROTATE_180)
    # height, width, _ = frame.shape

    # roi = frame[0:height, 0:width]

    class_index, confidence, bbox = model.detect(roi, confThreshold=0.64)
    print(class_index)

    if len(class_index) != 0:
        for (
            ClassInd,
            conf,
            boxes,
        ) in zip(class_index.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(roi, boxes, (255, 0, 0), 1)
                cv2.putText(
                    roi,
                    classLabels[ClassInd - 1],
                    (boxes[0] + 10, boxes[1] + 40),
                    font,
                    fontScale=font_scale,
                    color=(0, 255, 0),
                    thickness=3,
                )
                sound.play()

    cv2.imshow("image", roi)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
