from flask import Flask
from flask import render_template
from flask import Response
from imutils.video import VideoStream
import imutils
import cv2
import threading
import argparse
import time
import datetime


# setup flask
app = Flask(__name__)


# read the video stream
def video_frame(rotate, flip, enable_edges):
    # get global video stream, frame and lock
    global vs, outputFrame, lock

    # loop forever and read the current frame, resize and rotate
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        frame = imutils.rotate_bound(frame, rotate)
        if flip:
            frame = cv2.flip(frame, 1)

        if enable_edges:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blur, 50, 200)
            frame[edges == 255] = [255, 0, 0]

        # write the timestamp onto the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # get a lock and copy the current frame to the global frame
        with lock:
            outputFrame = frame.copy()


# encode the video frame to display on a web page
def encode_frame():
    # get global frame and lock
    global outputFrame, lock

    # loop forever, get the lock and if there is a frame encode it as JPEG
    # if the encoding failed then move on
    # yield the content type and encoded image as a byte string
    while True:
        with lock:
            if outputFrame is not None:
                (rc, encodedImage) = cv2.imencode(".jpg", outputFrame)
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
    __version__ = 0.1

    # pull in arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default="127.0.0.1", required=False, help="IP address of server")
    ap.add_argument("-p", "--port", type=int, default=8888, required=False, help="Port of server")
    ap.add_argument("-c", "--picam", action="store_true", default=False, required=False,
                    help="Enable Pi Camera")
    ap.add_argument("-r", "--rotate", type=int, default=0, required=False, help="Rotate image")
    ap.add_argument("-f", "--flip", action="store_true", default=False, required=False,
                    help="Flip image")
    ap.add_argument("-e", "--edges", action="store_true", default=False, required=False,
                    help="Enable edge detection")
    ap.add_argument("-v", "--version", action="version",
                    version="%(prog)s {version}".format(version=__version__))
    args = vars(ap.parse_args())

    # store the video frame and lock
    outputFrame = None
    lock = threading.Lock()

    # setup the video camera (switch between either pi camera or standard attached camera)
    if args["picam"]:
        vs = VideoStream(usePiCamera=True).start()
    else:
        vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # build a separate thread to manage the video stream
    thrd = threading.Thread(target=video_frame, args=(args["rotate"], args["flip"], args["edges"],))
    thrd.daemon = True
    thrd.start()

    # host / port - specify host and port arguments for the server
    # debug - enable debugging by default
    # threaded - each request is a handled by a separate thread
    # use_reloader - don't reload server should any module change (on by default if debug is true)
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

    vs.stop()
