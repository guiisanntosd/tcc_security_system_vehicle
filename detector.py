import cv2
import time
import numpy as np
from pygame import mixer

np.random.seed(20)


class Detector:
    def __init__(self, video, config, model, classes):
        self.video = video
        self.config = config
        self.model = model
        self.classes = classes

        self.net = cv2.dnn_DetectionModel(self.model, self.config)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def sound(self):
        mixer.init()
        mixer.Sound(r"./sounds/beep.wav").play()

    def readClasses(self):
        with open(self.classes, "rt") as f:
            self.classes = f.read().rstrip("\n").split("\n")

        self.color = np.random.uniform(low=0, high=255, size=(len(self.classes), 3))

    def onVideo(self):
        cap = cv2.VideoCapture(self.video)

        if cap.isOpened() == False:
            print("Error opening video stream or file")
            return

        ret, frame = cap.read()
        startTime = 0

        while ret:

            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime

            classIds, confs, bbox = self.net.detect(frame, confThreshold=0.5)

            bboxs = list(bbox)
            confidences = list(np.array(confs).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            indices = cv2.dnn.NMSBoxes(
                bboxs, confidences, score_threshold=0.5, nms_threshold=0.4
            )

            if len(indices) != 0:
                for i in range(0, len(indices)):
                    self.sound()
                    bbox = bboxs[np.squeeze(indices[i])]
                    classConfidence = confidences[np.squeeze(indices[i])]
                    classId = classIds[np.squeeze(indices[i])]
                    classLabel = self.classes[classId]
                    classColor = [int(c) for c in self.color[classId]]

                    displayText = "{}: {:.2f}".format(classLabel, classConfidence * 100)

                    x, y, w, h = bbox

                    cv2.rectangle(frame, (x, y), (x + w, y + h), classColor, 1)

                    cv2.putText(
                        frame,
                        displayText,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.5,
                        classColor,
                        2,
                    )

                    lineWidth = min(int(w * 0.3), int(h * 0.3))

                    cv2.line(frame, (x, y), (x + lineWidth, y), classColor, 10)
                    cv2.line(frame, (x, y), (x, y + lineWidth), classColor, 10)

                    cv2.line(frame, (x + w, y), (x + w - lineWidth, y), classColor, 10)
                    cv2.line(frame, (x + w, y), (x + w, y + lineWidth), classColor, 10)

                    cv2.line(frame, (x, y + h), (x + lineWidth, y + h), classColor, 10)
                    cv2.line(frame, (x, y + h), (x, y + h - lineWidth), classColor, 10)

                    cv2.line(
                        frame,
                        (x + w, y + h),
                        (x + w - lineWidth, y + h),
                        classColor,
                        10,
                    )
                    cv2.line(
                        frame,
                        (x + w, y + h),
                        (x + w, y + h - lineWidth),
                        classColor,
                        10,
                    )

            cv2.putText(
                frame,
                "FPS: {}".format(int(fps)),
                (10, 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Security System", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            _, frame = cap.read()

        cv2.destroyAllWindows()
