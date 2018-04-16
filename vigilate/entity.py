import random
from typing import List
from .coordinate import Coordinate


class Entity:
    def __init__(self, coordinate: Coordinate):
        self.history: List[Coordinate] = [coordinate]
        self.active = True
        self.color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))

    def update_position(self, coordinate: Coordinate):
        self.history.append(coordinate)

    def position(self):
        return self.history[-1]

    def positions(self):
        for position in self.history:
            yield position
