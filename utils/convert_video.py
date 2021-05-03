from config import version
from imutils import paths
import argparse
import os
import imageio


if __name__ == '__main__':
    __version__ = version.version

    # pull in arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="/tmp/", required=False,
                    help="Input path to the images")
    ap.add_argument("-o", "--output", type=str, default="/tmp/", required=False,
                    help="Output path for video file")
    ap.add_argument("-f", "--file", type=str, default="video.mp4", required=False,
                    help="Name of video file")
    ap.add_argument("-r", "--rate", type=int, default=30, required=False, help="Frames per second")
    ap.add_argument("-v", "--version", action="version",
                    version="%(prog)s {version}".format(version=__version__))
    args = vars(ap.parse_args())

    images = list(paths.list_images(args["input"]))
    filename = os.path.join(args["output"], args["file"])
    writer = imageio.get_writer(filename, fps=args["rate"])
    frames = len(images)

    for idx, image in enumerate(images):
        frame = idx + 1
        if frame % 100 == 0 or frame == frames:
            print("Writing image {} of {}".format(frame, frames))
        writer.append_data(imageio.imread(image))

writer.close()
