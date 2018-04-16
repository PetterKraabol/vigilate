from .entity import Entity
from .coordinate import Coordinate


class Line:
    def __init__(self, start: Coordinate, end: Coordinate, color):
        self.start: Coordinate = start
        self.end: Coordinate = end
        self.counter = 0
        self.color = color

    def passed_by(self, entity: Entity) -> bool:
        has_before: bool = False
        has_after: bool = False

        for position in entity.positions():
            if position.y < self.start.y:
                has_after = True
            if position.y > self.start.y:
                has_before = True

            if has_before and has_after:
                return True

        return False
