import logging
import logging.handlers
import os
import math
import sys
import cv2
import numpy as np

logging.disable(logging.DEBUG)

DISABLE_LOGGING = True

# ============================================================================

CAR_COLOURS = [(0, 0, 255), (0, 106, 255), (0, 216, 255), (0, 255, 182), (0, 255, 76)
    , (144, 255, 0), (255, 255, 0), (255, 148, 0), (255, 0, 178), (220, 0, 255)]


# ============================================================================

class Vehicle(object):
    def __init__(self, id, position):
        self.id = id
        self.positions = [position]
        self.frames_since_seen = 0
        self.counted = False

    @property
    def last_position(self):
        return self.positions[-1]

    def add_position(self, new_position):
        self.positions.append(new_position)
        self.frames_since_seen = 0

    def draw(self, output_image):
        car_colour = CAR_COLOURS[self.id % len(CAR_COLOURS)]
        for point in self.positions:
            cv2.circle(output_image, point, 2, car_colour, -1)
            cv2.polylines(output_image, [np.int32(self.positions)]
                          , False, car_colour, 1)


# ============================================================================

class VehicleCounter(object):
    def __init__(self, shape, divider):
        self.log = logging.getLogger("vehicle_counter")
        log.disabled = DISABLE_LOGGING

        self.height, self.width = shape
        self.divider = divider

        self.vehicles = []
        self.next_vehicle_id = 0
        self.vehicle_count = 0
        self.max_unseen_frames = 7

    @staticmethod
    def get_vector(a, b):
        """Calculate vector (distance, angle in degrees) from point a to point b.

        Angle ranges from -180 to 180 degrees.
        Vector with angle 0 points straight down on the image.
        Values increase in clockwise direction.
        """
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])

        distance = math.sqrt(dx ** 2 + dy ** 2)

        if dy > 0:
            angle = math.degrees(math.atan(-dx / dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx / dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx / dy))
            else:
                angle = 180.0

        return distance, angle

    @staticmethod
    def is_valid_vector(a):
        distance, angle = a
        threshold_distance = max(10.0, -0.008 * angle ** 2 + 0.4 * angle + 25.0)
        return distance <= threshold_distance

    def update_vehicle(self, vehicle, matches):
        # Find if any of the matches fits this vehicle
        for i, match in enumerate(matches):
            contour, centroid = match

            vector = self.get_vector(vehicle.last_position, centroid)
            if self.is_valid_vector(vector):
                vehicle.add_position(centroid)
                self.log.debug("Added match (%d, %d) to vehicle #%d. vector=(%0.2f,%0.2f)"
                               , centroid[0], centroid[1], vehicle.id, vector[0], vector[1])
                return i

        # No matches fit...
        vehicle.frames_since_seen += 1
        self.log.debug("No match for vehicle #%d. frames_since_seen=%d"
                       , vehicle.id, vehicle.frames_since_seen)

        return None

    def update_count(self, matches, output_image=None):
        self.log.debug("Updating count using %d matches...", len(matches))

        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches)
            if i is not None:
                del matches[i]

        # Add new vehicles based on the remaining matches
        for match in matches:
            contour, centroid = match
            new_vehicle = Vehicle(self.next_vehicle_id, centroid)
            self.next_vehicle_id += 1
            self.vehicles.append(new_vehicle)
            self.log.debug("Created new vehicle #%d from match (%d, %d)."
                           , new_vehicle.id, centroid[0], centroid[1])

        # Count any uncounted vehicles that are past the divider
        for vehicle in self.vehicles:
            if not vehicle.counted and (vehicle.last_position[1] > self.divider):
                self.vehicle_count += 1
                vehicle.counted = True
                self.log.debug("Counted vehicle #%d (total count=%d)."
                               , vehicle.id, self.vehicle_count)

        # Optionally draw the vehicles on an image
        if output_image is not None:
            for vehicle in self.vehicles:
                vehicle.draw(output_image)

            cv2.putText(output_image, ("%02d" % self.vehicle_count), (142, 10)
                        , cv2.FONT_HERSHEY_PLAIN, 0.7, (127, 255, 255), 1)

        # Remove vehicles that have not been seen long enough
        removed = [v.id for v in self.vehicles
                   if v.frames_since_seen >= self.max_unseen_frames]
        self.vehicles[:] = [v for v in self.vehicles
                            if not v.frames_since_seen >= self.max_unseen_frames]
        for vehicle_id in removed:
            self.log.debug("Removed vehicle #%d.", vehicle_id)

        self.log.debug("Count updated, tracking %d vehicles.", len(self.vehicles))


# ============================================================================

IMAGE_DIR = "images"
IMAGE_FILENAME_FORMAT = IMAGE_DIR + "/frame_%04d.png"

# Support either video file or individual frames
CAPTURE_FROM_VIDEO = True
if CAPTURE_FROM_VIDEO:
    IMAGE_SOURCE = "./media/regular.mp4"  # Video file
else:
    IMAGE_SOURCE = IMAGE_FILENAME_FORMAT  # Image sequence

# Time to wait between frames, 0=forever
WAIT_TIME = 1  # 250 # ms

LOG_TO_FILE = False

# Colours for drawing on processed frames
DIVIDER_COLOUR = (255, 255, 0)
BOUNDING_BOX_COLOUR = (255, 0, 0)
CENTROID_COLOUR = (0, 0, 255)


# ============================================================================

def init_logging():
    main_logger = logging.getLogger()
    main_logger.propagate = False

    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s'
        , datefmt='%Y-%m-%d %H:%M:%S')

    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    main_logger.addHandler(handler_stream)

    if LOG_TO_FILE:
        handler_file = logging.handlers.RotatingFileHandler("debug.log"
                                                            , maxBytes=2 ** 24
                                                            , backupCount=10)
        handler_file.setFormatter(formatter)
        main_logger.addHandler(handler_file)

    main_logger.setLevel(logging.DEBUG)

    return main_logger


# ============================================================================

def save_frame(file_name_format, frame_number, frame, label_format):
    file_name = file_name_format % frame_number
    label = label_format % frame_number

    log.debug("Saving %s as '%s'", label, file_name)
    cv2.imwrite(file_name, frame)


# ============================================================================

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return cx, cy


# ============================================================================

def detect_vehicles(fg_mask):
    log = logging.getLogger("detect_vehicles")
    log.disabled = DISABLE_LOGGING

    MIN_CONTOUR_WIDTH = 21
    MIN_CONTOUR_HEIGHT = 21

    # Find the contours of any vehicles in the image
    image, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    log.debug("Found %d vehicle contours.", len(contours))

    matches = []
    for (i, contour) in enumerate(contours):
        # print(contour)
        x, y, w, h = cv2.boundingRect(contour)
        contour_valid = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT)

        log.debug("Contour #%d: pos=(x=%d, y=%d) size=(w=%d, h=%d) valid=%s"
                  , i, x, y, w, h, contour_valid)

        if not contour_valid:
            continue

        centroid = get_centroid(x, y, w, h)

        matches.append(((x, y, w, h), centroid))

    return matches


# ============================================================================

def filter_mask(fg_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Fill any small holes
    closing = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=2)

    return dilation


# ============================================================================

def process_frame(frame_number, frame, bg_subtractor, car_counter):
    log = logging.getLogger("process_frame")
    log.disabled = DISABLE_LOGGING

    # Create a copy of source frame to draw into
    processed = frame.copy()

    # Draw dividing line -- we count cars as they cross this line.
    cv2.line(processed, (0, int(car_counter.divider)), (int(frame.shape[1]), int(car_counter.divider)), DIVIDER_COLOUR,
             1)

    # Remove the background
    fg_mask = bg_subtractor.apply(frame, None, 0.01)
    fg_mask = filter_mask(fg_mask)

    #save_frame(IMAGE_DIR + "/mask_%04d.png"
    #           , frame_number, fg_mask, "foreground mask for frame #%d")

    matches = detect_vehicles(fg_mask)

    log.debug("Found %d valid vehicle contours.", len(matches))
    for (i, match) in enumerate(matches):
        contour, centroid = match

        log.debug("Valid vehicle contour #%d: centroid=%s, bounding_box=%s", i, centroid, contour)

        x, y, w, h = contour

        # Mark the bounding box and the centroid on the processed frame
        # NB: Fixed the off-by one in the bottom right corner
        cv2.rectangle(processed, (x, y), (x + w - 1, y + h - 1), BOUNDING_BOX_COLOUR, 1)
        cv2.circle(processed, centroid, 2, CENTROID_COLOUR, -1)

    log.debug("Updating vehicle count...")
    car_counter.update_count(matches, processed)

    return processed


# ============================================================================

def main():
    log = logging.getLogger("main")
    log.disabled = DISABLE_LOGGING

    log.debug("Creating background subtractor...")
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    log.debug("Pre-training the background subtractor...")
    default_bg = cv2.imread(IMAGE_FILENAME_FORMAT % 119)
    bg_subtractor.apply(default_bg, None, 1.0)

    car_counter = None  # Will be created after first frame is captured

    # Set up image source
    log.debug("Initializing video capture device #%s...", IMAGE_SOURCE)
    cap = cv2.VideoCapture(IMAGE_SOURCE)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    log.debug("Video capture frame size=(w=%d, h=%d)", frame_width, frame_height)

    log.debug("Starting capture loop...")
    frame_number = -1
    while True:
        frame_number += 1
        log.debug("Capturing frame #%d...", frame_number)
        ret, frame = cap.read()
        if not ret:
            log.error("Frame capture failed, stopping...")
            break

        log.debug("Got frame #%d: shape=%s", frame_number, frame.shape)

        if car_counter is None:
            # We do this here, so that we can initialize with actual frame size
            log.debug("Creating vehicle counter...")
            car_counter = VehicleCounter(frame.shape[:2], frame.shape[0] / 2)

        # Archive raw frames from video to disk for later inspection/testing
        if CAPTURE_FROM_VIDEO:
            save_frame(IMAGE_FILENAME_FORMAT
                       , frame_number, frame, "source frame #%d")

        log.debug("Processing frame #%d...", frame_number)
        processed = process_frame(frame_number, frame, bg_subtractor, car_counter)

        save_frame(IMAGE_DIR + "/processed_%04d.png"
                   , frame_number, processed, "processed frame #%d")

        cv2.imshow('Source Image', frame)
        cv2.imshow('Processed Image', processed)

        log.debug("Frame #%d processed.", frame_number)

        c = cv2.waitKey(WAIT_TIME)
        if c == 27:
            log.debug("ESC detected, stopping...")
            break

    log.debug("Closing video capture device...")
    cap.release()
    cv2.destroyAllWindows()
    log.debug("Done.")


# ============================================================================

if __name__ == "__main__":
    log = init_logging()
    log.disabled = DISABLE_LOGGING

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
