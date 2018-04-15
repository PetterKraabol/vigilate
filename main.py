import argparse
import subprocess
from devices import RaspberryPi


def main(arguments: dict):
    device: RaspberryPi = RaspberryPi(arguments['name'],
                                      serial_port=arguments['serial_port'],
                                      baud_rate=arguments['baud_rate'])
    device.lorawan.transmit(b'Hello world')
    print(device.lorawan.receive())


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description='Vigilate RaspberryPi')
    parser.add_argument('-n', '--name', type=str, help='Device name')
    parser.add_argument('--serial', type=str, default='/dev/serial0', help='Serial port')
    parser.add_argument('--baud_rate', type=int, default=115200, help='Serial baud rate')
    return parser.parse_args().__dict__


if __name__ == '__main__':
    label = subprocess.check_output(["git", "describe"]).strip()

    # Run main program
    main(parse_arguments())
ø