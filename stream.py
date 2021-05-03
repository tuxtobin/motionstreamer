from config import version
from flask import Flask
from flask import render_template
from flask import Response
from imutils.video import VideoStream
from helpers.motion_detector import MotionDetector
import imutils
import cv2
import threading
import argparse
import time
import datetime
import os
import configparser


# setup flask
app = Flask(__name__)


# read the video stream
def video_frame(rotate, flip, snapshot, output, bg_frames):
    # get global video stream, frame and lock
    global vs, currentFrame, lock

    # instantiate motion detector (using background subtraction)
    md = MotionDetector(accumWeight=0.1)
    # initialise number of accumulated background frames
    total_bg_frames = 0

    # loop forever and read the current frame, resize and rotate
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=640, inter=cv2.INTER_NEAREST)
        frame = imutils.rotate_bound(frame, rotate)

        if flip:
            frame = cv2.flip(frame, 1)

        # convert to gray scale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # write the timestamp onto the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime("%a %d %B %Y %H:%M:%S"), (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # if there are sufficient frames start looking for motion
        if total_bg_frames > bg_frames:
            # check for motion
            motion = md.detect(gray)

            # if there was sufficient change then draw the bounding box around it
            # save the image
            if motion is not None:
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 255, 0), 1)
                path = os.path.join(output, timestamp.strftime("%Y-%m-%d"))
                if not os.path.isdir(path):
                    os.mkdir(path)
                filename = os.path.join(path, timestamp.strftime("%H-%M-%S") + ".jpg")
                cv2.imwrite(filename, frame)

        # update the background
        md.update(gray)
        total_bg_frames += 1

        # get a lock and copy the current frame to the global frame
        with lock:
            currentFrame = frame.copy()

        # save frame each N seconds (only if not using motion detection)
        if snapshot > 0:
            path = os.path.join(output, timestamp.strftime("%Y-%m-%d"))
            if not os.path.isdir(path):
                os.mkdir(path)
            filename = os.path.join(path, timestamp.strftime("%H-%M-%S") + ".jpg")
            cv2.imwrite(filename, frame)
            time.sleep(snapshot)


# encode the video frame to display on a web page
def encode_frame():
    # get global frame and lock
    global currentFrame, lock

    # loop forever, get the lock and if there is a frame encode it as JPEG
    # if the encoding failed then move on
    # yield the content type and encoded image as a byte string
    while True:
        with lock:
            if currentFrame is not None:
                (rc, encodedImage) = cv2.imencode(".jpg", currentFrame)
                if not rc:
                    continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(encode_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    __version__ = version.version

    # pull in arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--version", action="version",
                    version="%(prog)s {version}".format(version=__version__))
    args = vars(ap.parse_args())

    # read in configuration settings
    config = configparser.ConfigParser()
    config.read("config\\settings.ini")
    cfg = dict(config.items("MAIN"))

    # store the video frame and lock
    currentFrame = None
    lock = threading.Lock()

    # if motion detection is enabled then disable snapshot processing
    if cfg["detect"]:
        cfg["snapshot"] = 0
    else:
        if "snapshot" not in cfg or not isinstance(cfg["snapshot"], int):
            cfg["snapshot"] = 0

    # setup the video camera (switch between either pi camera or standard attached camera)
    if cfg["picam"]:
        vs = VideoStream(usePiCamera=True).start()
    else:
        vs = VideoStream(src=0).start()
    time.sleep(2.0)
    prevFrame = vs.read()

    # build a separate thread to manage the video stream
    thrd = threading.Thread(target=video_frame, args=(cfg["rotate"], cfg["flip"],
                                                      cfg["snapshot"], cfg["output"],
                                                      cfg["bg_frames"],))
    thrd.daemon = True
    thrd.start()

    # host / port - specify host and port arguments for the server
    # debug - enable debugging by default
    # threaded - each request is a handled by a separate thread
    # use_reloader - don't reload server should any module change (on by default if debug is true)
    app.run(host=cfg["ip"], port=cfg["port"], debug=True, threaded=True, use_reloader=False)

    vs.stop()
