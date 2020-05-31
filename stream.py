import argparse


__version__ = 0.1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default="127.0.0.1", required=False, help="IP address of server")
    ap.add_argument("-p", "--port", type=int, default=8888, required=False, help="Port of server")
    ap.add_argument("-r", "--rotate", type=int, default=0, required=False, help="Rotate image")
    ap.add_argument("-e", "--edges", action="store_true", default=False, required=False,
                    help="Turn on edge detection")
    ap.add_argument("-v", "--version", action="version",
                    version="%(prog)s {version}".format(version=__version__))
    args = vars(ap.parse_args())
