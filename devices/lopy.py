import serial


class LoPy:
    def __init__(self, port: str, baud_rate: int = 115200, timeout: int = None):
        self.serial: serial.Serial = serial.Serial(port=port, baudrate=baud_rate, timeout=timeout)
