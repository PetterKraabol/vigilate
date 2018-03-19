import cv2
import time
from sensors import Camera, GPS
from devices import LoPy
from vigilate import Video


def main():
    lopy: LoPy = LoPy('COM4')
    for i in range(10):
        print('Reading...')
        message: str = str(lopy.serial.read(9))
        print(message)
        time.sleep(1)


if __name__ == '__main__':
    print(f'Currently using OpenCV {cv2.__version__}')
    main()
