import numpy as np
import imutils
import cv2


class MotionDetector:
    def __init__(self, accum_weight=0.5):
        self.accum_weight = accum_weight
        self.bg = None

    def update(self, image):
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        cv2.accumulateWeighted(image, self.bg, self.accum_weight)

    def detect(self, image, tval=25):
        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        thresh = cv2.threshold(delta, tval, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        if len(cnts) == 0:
            return None

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            (minX, minY) = (min(minX, x), min(minY, y))
            (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

        return thresh, (minX, minY, maxX, maxY)
