import argparse
from typing import Union

from vigilate import Vigilate
from vigilate import VigilateHorizontal


def main(arguments: dict):
    # device: RaspberryPi = RaspberryPi(arguments['name'],
    #                                  serial_port=arguments['serial_port'],
    #                                  baud_rate=arguments['baud_rate'])

    program: Union[VigilateHorizontal, Vigilate] = VigilateHorizontal() if arguments['horizontal'] else Vigilate()
    program.start()


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description='Vigilate RaspberryPi')
    parser.add_argument('-n', '--name', type=str, help='Device name')
    parser.add_argument('--serial', type=str, default='/dev/serial0', help='Serial port')
    parser.add_argument('--horizontal', action='store_true', help='Serial port')
    parser.add_argument('--baud_rate', type=int, default=115200, help='Serial baud rate')
    return parser.parse_args().__dict__


if __name__ == '__main__':
    main(parse_arguments())
