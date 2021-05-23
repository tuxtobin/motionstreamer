from collections import deque
from threading import Thread
from queue import Queue
import time
import cv2


class BufferedFrame:
    def __init__(self, buffer_size=64, timeout=1.0):
        self.bufferSize = buffer_size
        self.timeout = timeout
        self.output = ""
        self.counter = -1
        self.frames = deque(maxlen=buffer_size)
        self.queue = None
        self.thread = None
        self.recording = False

    def update(self, frame):
        self.frames.appendleft(frame)

        if self.recording:
            self.queue.put(frame)

    def start(self, output):
        self.recording = True
        self.queue = Queue()
        self.output = output

        for i in range(len(self.frames), 0, -1):
            self.queue.put(self.frames[i - 1])

        self.thread = Thread(target=self.write, args=())
        self.thread.daemon = True
        self.thread.start()

    def write(self):
        while True:
            if not self.recording:
                return

            if not self.queue.empty():
                frame = self.queue.get()
                self.counter += 1
                filename = self.output + "_" + str(self.counter) + ".jpg"
                cv2.imwrite(filename, frame)
            else:
                time.sleep(self.timeout)

    def flush(self):
        while not self.queue.empty():
            frame = self.queue.get()
            self.counter += 1
            filename = self.output + "_" + str(self.counter) + ".jpg"
            cv2.imwrite(filename, frame)

    def finish(self):
        self.recording = False
        self.thread.join()
        self.flush()
        self.counter = -1
