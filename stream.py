from config import version
from flask import Flask
from flask import render_template
from flask import Response
from imutils.video import VideoStream
from helpers.motion_detector import MotionDetector
from helpers.buffered_frame import BufferedFrame
import imutils
import cv2
import threading
import argparse
import time
import datetime
import os
import signal
import sys
import configparser


# setup flask
app = Flask(__name__)


# signal detector
def signal_handler(sig, frame):
    if bf.recording:
        bf.finish()

    vs.stop()
    sys.exit(0)


# read configuration file
def read_config(config_file):
    config = {}
    cp = configparser.ConfigParser()
    cp.read(config_file)
    for section in cp.sections():
        for key in cp[section]:
            config[key] = cp[section][key]

    return config


# detector - read the video stream
def detector_video_frame(rotate, flip, output, background, buffer_size, min_area, hidden_area):
    # get global video stream, frame and lock
    global vs, currentFrame, lock

    # initialise the buffered frame class and number of continuous frames
    bf = BufferedFrame(buffer_size)
    cont_frames = 0
    # instantiate motion detector (using background subtraction)
    md = MotionDetector(accum_weight=0.1, x1=hidden_area[0], y1=hidden_area[1],
                        x2=hidden_area[2], y2=hidden_area[3])
    # initialise number of accumulated background frames
    total_bg_frames = 0

    # loop forever and read the current frame, resize and rotate
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=640, inter=cv2.INTER_NEAREST)
        frame = imutils.rotate_bound(frame, rotate)

        # initialise flag to update the continuous frame counter
        update_cont_frames = True

        # flip frame if requested
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
        if total_bg_frames > background:
            # check for motion
            motion = md.detect(gray)

            # if there was sufficient change then draw the bounding box around it
            # save the image
            if motion is not None:
                (thresh, (minX, minY, maxX, maxY)) = motion
                # check if the motion meets the minimum area
                update_cont_frames = ((maxX - minX) * (maxY - minY)) >= min_area
                if update_cont_frames:
                    # reset the period of motion
                    cont_frames = 0
                    # draw a rectangle around the area of disturbance
                    cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 255, 0), 1)

                    # if not recording, then start
                    if not bf.recording:
                        path = os.path.join(output, timestamp.strftime("%Y-%m-%d"))
                        if not os.path.isdir(path):
                            os.mkdir(path)
                        filename = os.path.join(path, timestamp.strftime("%H-%M-%S") + ".avi")
                        bf.start(filename, cv2.VideoWriter_fourcc(*'MJPG'), 32)

        # if there had been movement increment continuous frame counter
        if update_cont_frames:
            cont_frames += 1

        # update frame buffer
        bf.update(frame)

        # if still recording and the continuous frames meets the buffer size then stop
        if bf.recording and cont_frames == buffer_size:
            bf.finish()

        # update the background
        md.update(gray)
        total_bg_frames += 1

        # get a lock and copy the current frame to the global frame
        with lock:
            currentFrame = frame.copy()


# snapshot - read the video stream
def snapshot_video_frame(rotate, flip, output, frequency):
    # get global video stream, frame and lock
    global vs, currentFrame, lock

    # loop forever and read the current frame, resize and rotate
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=640, inter=cv2.INTER_NEAREST)
        frame = imutils.rotate_bound(frame, rotate)

        if flip:
            frame = cv2.flip(frame, 1)

        # write the timestamp onto the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime("%a %d %B %Y %H:%M:%S"), (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # save frame each N seconds (only if not using motion detection)
        if frequency > 0:
            path = os.path.join(output, timestamp.strftime("%Y-%m-%d"))
            if not os.path.isdir(path):
                os.mkdir(path)
            filename = os.path.join(path, timestamp.strftime("%H-%M-%S") + ".jpg")
            cv2.imwrite(filename, frame)
            time.sleep(frequency)

        # get a lock and copy the current frame to the global frame
        with lock:
            currentFrame = frame.copy()


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
    ap.add_argument("-i", "--ip", type=str, default="127.0.0.1", required=False, help="IP address of server")
    ap.add_argument("-p", "--port", type=int, default=8888, required=False, help="Port of server")
    ap.add_argument("-c", "--picam", action="store_true", default=False, required=False,
                    help="Enable Pi Camera")
    ap.add_argument("-r", "--rotate", type=int, default=0, required=False, help="Rotate image")
    ap.add_argument("-f", "--flip", action="store_true", default=False, required=False,
                    help="Flip image")
    ap.add_argument("-v", "--version", action="version",
                    version="%(prog)s {version}".format(version=__version__))
    ap.add_argument("-o", "--output", type=str, default="/tmp/", required=False,
                    help="Stop frame output path")
    sp = ap.add_subparsers(dest="subcommand")
    sp1 = sp.add_parser("snapshot", help="Take a snapshot every N second")
    sp1.add_argument("-q", "--frequency", type=int, default=0, required=False,
                     help="Frequency of snapshots (default is off")
    sp2 = sp.add_parser("detect", help="Enable motion detection")
    sp2.add_argument("-b", "--background", type=int, default=32, required=False, help="Number of background frames")
    sp2.add_argument("-s", "--buffer", type=int, default=64, required=False, help="Number of frames to buffer")
    sp2.add_argument("-a", "--area", type=int, default=10000, required=False, help="Minimum area for motion event")
    sp2.add_argument("-h", "--hidden", type=int, nargs='+', required=False, help="Area to hidden from motion detection")
    args = vars(ap.parse_args())
    args["hidden"] = tuple(args["hidden"])

    # begin capturing signals
    signal.signal(signal.SIGINT, signal_handler)

    # store the video frame and lock
    currentFrame = None
    lock = threading.Lock()

    # setup the video camera (switch between either pi camera or standard attached camera)
    if args["picam"]:
        vs = VideoStream(usePiCamera=True).start()
    else:
        vs = VideoStream(src=0).start()
    time.sleep(2.0)
    prevFrame = vs.read()

    # build a separate thread to manage the video streams
    if args["subcommand"] == "snapshot":
        thrd = threading.Thread(target=snapshot_video_frame, args=(args["rotate"], args["flip"],
                                                                   args["output"], args["frequency"],))
    elif args["subcommand"] == "detect":
        thrd = threading.Thread(target=detector_video_frame, args=(args["rotate"], args["flip"],
                                                                   args["output"], args["background"],
                                                                   args["buffer"], args["area"],
                                                                   args["hidden"],))

    thrd.daemon = True
    thrd.start()

    # host / port - specify host and port arguments for the server
    # debug - enable debugging by default
    # threaded - each request is a handled by a separate thread
    # use_reloader - don't reload server should any module change (on by default if debug is true)
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)
