import os
from detector import *


def main():
    video = r"./videos/tcc.mp4"
    config = os.path.join("config", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    model = os.path.join("config", "frozen_inference_graph.pb")
    classes = os.path.join("config", "coco.names")

    detector = Detector(video, config, model, classes)
    detector.onVideo()


if __name__ == "__main__":
    main()
