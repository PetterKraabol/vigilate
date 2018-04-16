import cv2


class Camera(cv2.VideoCapture):
    def __init__(self, *args, **kwargs):
        super(Camera, self).__init__(*args, **kwargs)

    def stream(self):
        while True:
            success, frame = self.read()
            if success:
                yield frame
            else:
                print('No frames')
                break

        self.release()
