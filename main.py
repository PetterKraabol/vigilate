import cv2
from sensors import Camera, GPS
from vigilate import Video


def main():
    video: Video = Video()

    for frame in video.file_stream('media/regular.mp4'):
        pass


if __name__ == '__main__':
    print(f'Currently using OpenCV {cv2.__version__}')
    main()
