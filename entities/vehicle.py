class Vehicle:
    def __init__(self, identity: int, position):
        self.identity = identity
        self.position = position

    @property
    def last_position(self):
        return 0
