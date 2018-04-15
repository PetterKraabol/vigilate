import time
import serial
from typing import Union, List


class LoPy:
    def __init__(self, port: str = '/dev/serial0', baud_rate: int = 115200, timeout: int = 10):
        self.serial = serial.Serial(port, baud_rate, timeout=timeout)

    def transmit(self, data: Union[bytes, str]):

        # Split data into 255 byte segments for sending
        segments: List[bytes] = [data[i:i+254] for i in range(0, len(data), 254)]

        for segment in segments:
            self.serial.write(bytes(segment, 'utf-8'))

    def receive(self) -> str:
        while True:
            if self.serial.in_waiting > 0:
                received = self.serial.read(self.serial.in_waiting).decode('utf-8')
                return received
            time.sleep(1)
