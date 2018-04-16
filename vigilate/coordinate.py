class Coordinate:
    def __init__(self, x: int = None, y: int = None):
        self.x = x
        self.y = y

    def __repr__(self):
        return self.x, self.y

    def __str__(self):
        return f'{self.x}, {self.y}'
