from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2
import numpy as np
from telegram_utils import send_telegram
import datetime
import threading


def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    #print(polygon.contains(centroid))
    return polygon.contains(centroid)


def cal_position(detections, i, cols, rows, frame):
    # Lay class_id
    class_id = int(detections[0, 0, i, 1])

    # Tinh toan vi tri cua doi tuong
    xLeftBottom = int(detections[0, 0, i, 3] * 300)
    yLeftBottom = int(detections[0, 0, i, 4] * 300)
    xRightTop = int(detections[0, 0, i, 5] * 300)
    yRightTop = int(detections[0, 0, i, 6] * 300)

    heightFactor = frame.shape[0] / 300.0  # frame cua cap.read
    widthFactor = frame.shape[1] / 300.0

    xLeftBottom = int(widthFactor * xLeftBottom)
    yLeftBottom = int(heightFactor * yLeftBottom)
    xRightTop = int(widthFactor * xRightTop)
    yRightTop = int(heightFactor * yRightTop)

    return  xLeftBottom, yLeftBottom, xRightTop, yRightTop


class YoloDetect():
    def __init__(self, detect_class="person", frame_width=1280, frame_height=720):
        # Parameters
        self.classnames_file = "model/classnames.txt"
        self.weights_file = "../MiAI_MobileNetSSD_Pi/MobileNetSSD_deploy.caffemodel"
        self.config_file = "../MiAI_MobileNetSSD_Pi/MobileNetSSD_deploy.prototxt"
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.detect_class = detect_class
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scale = 1 / 255
        self.model = cv2.dnn.readNetFromCaffe(self.config_file, self.weights_file)
        self.classes = {0: 'background',
                        1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                        5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                        10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                        14: 'motorbike', 15: 'person', 16: 'pottedplant',
                        17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
        self.output_layers = None
        # self.read_class_file()
        self.get_output_layers()
        self.last_alert = None
        self.alert_telegram_each = 15  # seconds

    def read_class_file(self):
        with open(self.classnames_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def get_output_layers(self):
        layer_names = self.model.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]

    def draw_prediction(self, img, class_id, x, y, x_plus_w, y_plus_h, points, confedence):
        label = str(self.classes[class_id]) + ":" + str(confedence)
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Tinh toan centroid
        centroid = ((x + x_plus_w) // 2, (y + y_plus_h) // 2)
        cv2.circle(img, centroid, 5, (color), -1)

        if isInside(points, centroid):
            img = self.alert(img)

        return isInside(points, centroid)

    def alert(self, img):
        cv2.putText(img, "ALARM!!!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # New thread to send telegram after 15 seconds
        if (self.last_alert is None) or (
                (datetime.datetime.utcnow() - self.last_alert).total_seconds() > self.alert_telegram_each):
            self.last_alert = datetime.datetime.utcnow()
            cv2.imwrite("alert.png", cv2.resize(img, dsize=None, fx=0.2, fy=0.2))
            thread = threading.Thread(target=send_telegram)
            thread.start()
        return img

    def detect(self, frame, points):
        class_ids = []
        confidences = []
        boxes = []


        blob = cv2.dnn.blobFromImage(frame, self.scale, (300,300 ), (0, 0, 0), True, crop=False)
        self.model.setInput(blob)
        detections = self.model.forward()

        cols = frame.shape[1]
        rows = frame.shape[0]

        # Duyet qua cac object detect duoc, output boundingbox = [[[[m, classID, confidence, xleftBotom, yLeftBottom,
        # xrightop, yrightTop]]]]
        for i in range(detections.shape[2]):
            # Lay gia tri confidence
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            # Neu vuot qua 0.5 threshold
            if (confidence >= self.conf_threshold) and (self.classes[class_id] == self.detect_class):
                # Tinh toan vi tri cua doi tuong
                x_left_top, y_left_top, x_right_bottom, y_right_bottom = cal_position(detections, i,cols, rows, frame)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x_left_top, y_left_top, x_right_bottom, y_right_bottom])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

            for i in indices:
                box = boxes[i]
                x = box[0]
                y = box[1]
                xb = box[2]
                yb = box[3]
                self.draw_prediction(frame, class_ids[i], x, y , xb, yb, points, confidences[i])

        return frame
