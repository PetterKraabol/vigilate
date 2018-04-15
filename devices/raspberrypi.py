import uuid
from devices import LoPy
from sensors import Camera, GPS


class RaspberryPi:
    def __init__(self, name: str = str(uuid.getnode()), serial_port: str ='/dev/serial0', baud_rate: int = 115200):
        self.name: str = name
        self.lorawan = LoPy(port=serial_port, baud_rate=baud_rate, timeout=10)
        self.camera: Camera = Camera()
        self.gps: GPS = GPS()

    def calibrate(self):
        pass

    def run(self):
        pass

    def shutdown(self):
        pass

    def restart(self):
        pass

    def status(self):
        pass
