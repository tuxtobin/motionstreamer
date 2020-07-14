from config import settings as config
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
import os


# setup flask
app = Flask(__name__)


# read the video stream
def video_frame(rotate, flip, enable_edges, enable_diff, stopframe, output, fd):
    # get global video stream, frame and lock
    global vs, prevFrame, currentFrame, lock, writeFlag

    # loop forever and read the current frame, resize and rotate
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=640, inter=cv2.INTER_NEAREST)
        frame = imutils.rotate_bound(frame, rotate)

        if flip:
            frame = cv2.flip(frame, 1)

        if enable_diff:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            frame2 = imutils.resize(prevFrame, width=400, inter=cv2.INTER_NEAREST)
            frame2 = imutils.rotate_bound(frame2, rotate)

            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

            frame_delta = cv2.absdiff(gray, gray2)

        if enable_edges:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            edges = cv2.Canny(gray, 50, 200)
            frame[edges == 255] = [255, 0, 0]

        # write the timestamp onto the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime("%a %d %B %Y %H:%M:%S"), (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # get a lock and copy the current frame to the global frame
        with lock:
            currentFrame = frame.copy()
            prevFrame = currentFrame

        # save frame each N seconds
        if stopframe > 0:
            path = os.path.join(output, timestamp.strftime("%Y-%m-%d"))
            if not os.path.isdir(path):
                os.mkdir(path)
            filename = os.path.join(path, timestamp.strftime("%H-%M-%S") + ".jpg")
            cv2.imwrite(filename, frame)
            time.sleep(stopframe)

            # write data to output device each 30 minutes
            if fd is not None:
                if int(timestamp.strftime('%M')) == 0 or int(timestamp.strftime('%M')) == 30:
                    if not writeFlag:
                        os.fsync(fd)
                        writeFlag = True
                else:
                    if writeFlag:
                        writeFlag = False


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
    __version__ = config.version

    # pull in arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default="127.0.0.1", required=False, help="IP address of server")
    ap.add_argument("-p", "--port", type=int, default=8888, required=False, help="Port of server")
    ap.add_argument("-c", "--picam", action="store_true", default=False, required=False,
                    help="Enable Pi Camera")
    ap.add_argument("-r", "--rotate", type=int, default=0, required=False, help="Rotate image")
    ap.add_argument("-f", "--flip", action="store_true", default=False, required=False,
                    help="Flip image")
    ap.add_argument("-v", "--version", action="version",
                    version="%(prog)s {version}".format(version=__version__))
    ap.add_argument("-s", "--stopframe", type=int, default=0, required=False,
                    help="Stop frame capture every N second")
    ap.add_argument("-o", "--output", type=str, default="/tmp/", required=False,
                    help="Stop frame output path")
    ap.add_argument("-w", "--write", action="store_true", default=False, required=False,
                    help="Destage the output directory")
    sp = ap.add_mutually_exclusive_group()
    sp.add_argument("-e", "--edges", action="store_true", default=False, required=False,
                    help="Enable edge detection")
    sp.add_argument("-d", "--diff", action="store_true", default=False, required=False,
                    help="Detection difference between frames")
    args = vars(ap.parse_args())

    # store the video frame and lock
    prevFrame = None
    currentFrame = None
    writeFlag = False
    lock = threading.Lock()
    fd = None

    # open the output directory and store the file description for later
    if args["write"]:
        fd = os.open(args["output"], os.O_DIRECTORY)

    # setup the video camera (switch between either pi camera or standard attached camera)
    if args["picam"]:
        vs = VideoStream(usePiCamera=True).start()
    else:
        vs = VideoStream(src=0).start()
    time.sleep(2.0)
    prevFrame = vs.read()

    # build a separate thread to manage the video stream
    thrd = threading.Thread(target=video_frame, args=(args["rotate"], args["flip"],
                                                      args["edges"], args["diff"],
                                                      args["stopframe"], args["output"],
                                                      fd,))
    thrd.daemon = True
    thrd.start()

    # host / port - specify host and port arguments for the server
    # debug - enable debugging by default
    # threaded - each request is a handled by a separate thread
    # use_reloader - don't reload server should any module change (on by default if debug is true)
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

    vs.stop()
