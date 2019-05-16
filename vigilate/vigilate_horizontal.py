import cv2
import math
from sensors import Camera, GPS
from typing import List, Tuple
from devices import RaspberryPi

from .coordinate import Coordinate
from .entity import Entity
from .line import Line


class VigilateHorizontal:
    def __init__(self, pi: RaspberryPi = None, entrance_line: int = 800, exit_line: int = 960, preview: bool = True):
        self.entrance = Coordinate(entrance_line, 0)
        self.exit = Coordinate(exit_line, 0)
        self.line_thickness = 1  # If object is within n pixels of line, increase counter

        # Sensors
        # self.camera = Camera(0) # Webcam
        self.camera = Camera('media/horizontal.mp4')
        # self.camera = Camera('media/complex.mp4')

        # Devices
        self.pi: RaspberryPi = pi

        # Settings
        self.width = 1920
        self.height = 1080
        self.camera.set(3, self.width)
        self.camera.set(4, self.height)
        self.reference_frame = None
        self.entities: List[Entity] = []
        self.entity_max_radius: int = 100
        self.preview: bool = preview
        self.lines: List[Line] = []

        self.entrance_counter = 0
        self.exit_counter = 0
        self.min_contour_area = 5000
        self.max_contour_area = 200000
        self.binarization_threshold = 70

        # Lines
        #self.add_line(Coordinate(1000, 1000), Coordinate(self.width, 1000), (0, 255, 255))

    @staticmethod
    def distance(a: Coordinate, b: Coordinate) -> float:
        return math.sqrt(math.pow(a.x - b.x, 2) + math.pow(a.y - b.y, 2))

    def add_line(self, start: Coordinate, end: Coordinate, color):
        self.lines.append(Line(start, end, color))

    def active_entities(self):
        for entity in self.entities:
            if entity.active:
                yield entity

    def at_entrance(self, entity):
        return abs(entity.position().y - self.entrance.y) <= self.line_thickness

    def at_exit(self, entity):
        return abs(entity.position().x - self.exit.x) <= self.line_thickness

    @staticmethod
    def passed_line(line: Coordinate, entity: Entity) -> bool:
        has_before: bool = False
        has_after: bool = False

        for position in entity.positions():
            if position.x < line.x:
                has_after = True
            if position.x > line.x:
                has_before = True

            if has_before and has_after:
                return True

        return False

    @staticmethod
    def gray_scale(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        return frame

    def new_or_nearest(self, position: Coordinate) -> Entity:
        nearest: Tuple[Entity, float] = (None, self.entity_max_radius + 1)

        # Find closest entity (if any)
        for entity in self.active_entities():
            distance = self.distance(position, entity.position())
            if distance <= self.entity_max_radius and distance < nearest[1]:
                nearest = (entity, distance)

        if nearest[0] is None:
            entity = Entity(position)
            self.entities.append(entity)
            return entity
        else:
            # Existing entity found. Update position
            nearest[0].update_position(position)
            return nearest[0]

    @staticmethod
    def filter_mask(fg_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Fill any small holes
        closing = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        # Remove noise
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # Dilate to merge adjacent blobs
        dilation = cv2.dilate(opening, kernel, iterations=2)

        return dilation

    def transmit_data(self):
        data: str = f'{self.exit_counter},{self.pi.gps.location()}'
        self.pi.lorawan.transmit(bytes(data))

    def start(self):
        # Skip first frames to let camera calibrate itself
        #for i in range(20):
        #    self.camera.read()

        # Subtractor
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        # for i in range(300):
        #    _, image = self.camera.read()
        #    bg_subtractor.apply(image, None, 0.01)

        # Get frames from camera stream
        for i, frame in enumerate(self.camera.stream()):

            # Transmit data over LoRaWAN every 900th frame
            if i % 900 == 0 and self.pi is not None:
                self.transmit_data()

            # Gray scale
            # frame_gray = bg_subtractor.apply(frame, None, 0.01)

            # Set reference frame if none
            # if self.reference_frame is None:
            #    self.reference_frame = frame_gray
            #    continue

            # Subtract reference frame from image
            # frame_delta = cv2.absdiff(self.reference_frame, frame_gray)
            # _, frame_threshold = cv2.threshold(frame_delta, self.binarization_threshold, 255, cv2.THRESH_BINARY)

            # Dilate image
            # frame_threshold = cv2.dilate(frame_threshold, None, iterations=2)

            fg_mask = bg_subtractor.apply(frame, None, 0.01)
            fg_mask = VigilateHorizontal.filter_mask(fg_mask)

            # Find contours (objects)
            contours, _ = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Display foreground masking frame
            #frame = fg_mask

            # Plot entrance and exit lines
            # cv2.line(frame, (0, self.entrance.y), (self.width, self.entrance.y), (255, 0, 0), self.line_thickness)
            cv2.line(frame, (self.exit.x, 0), (self.exit.x, self.height), (255, 0, 255), self.line_thickness)

            # Check contours
            for contour in contours:
                # Rectangle information
                x, y, width, height = cv2.boundingRect(contour)

                # Ignore contours which are either too small or too large.
                if cv2.contourArea(contour) < self.min_contour_area or cv2.contourArea(contour) > self.max_contour_area:
                    continue

                # Find object centroid
                centroid = Coordinate(int((x + x + width) / 2), int((y + y + height) / 2))

                # Create or find existing entity
                entity: Entity = self.new_or_nearest(centroid)

                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + width, y + width), entity.color, 2)

                if VigilateHorizontal.passed_line(self.exit, entity):
                    entity.active = False
                    self.exit_counter += 1
                    continue

                # Draw history line
                previous_position: Coordinate = None
                for position in entity.positions():
                    # First position
                    if previous_position is None:
                        previous_position = position
                        continue

                    cv2.line(frame, (previous_position.x, previous_position.y), (position.x, position.y), entity.color,
                             1)
                    cv2.circle(frame, (position.x, position.y), 5, entity.color, 2)
                    previous_position = position

            # Write stats on screen
            # cv2.putText(frame, f'Entrances: {self.entrance_counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #            (250, 0, 1), 2)
            cv2.putText(frame, f'Counter: {self.exit_counter}', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        color=(255, 0, 0), thickness=2)

            # Display frame preview
            if self.preview:
                cv2.imshow('Monitor', frame)
                cv2.waitKey(1)

        # Cleanup
        cv2.destroyAllWindows()
