import cv2
import numpy
from typing import Generator
from sensors import Camera


class Video:
    def __init__(self):
        self.camera: Camera = Camera

    def stream(self):
        pass

    @staticmethod
    def file_stream(filename: str) -> Generator[numpy.ndarray, None, None]:
        video: cv2.VideoCapture = cv2.VideoCapture(filename)
        while True:
            active, image = video.read()
            if not active:
                break
            yield image