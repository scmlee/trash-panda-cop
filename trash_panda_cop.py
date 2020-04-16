import cv2
import imutils
import datetime
import time
import platform
import config
from logzero import logger, setup_logger, logging
from pathlib import Path
#import azure_raccoon_recognizer  # as raccoon_recognizer
from raccoon_recognizer.racoon_recognizer import RacoonRecognizer
from video_stream.video_stream import VideoStream

# Constants / Parameters
_CONTOUR_TOLERANCE = 8000
_MOTION_ELAPSED_TIME_TOLERANCE = 3.0
_CAPTURE_FPS = 10
_SHOW_PREVIEW = True
_DEBUG = True
_RESULTS_PATH = Path("results")

def initializeWindow(*windows):
    ''' Initialize a set of cv2 named windows '''
    if _SHOW_PREVIEW:
        for window in windows:
            cv2.namedWindow(window)


def destroyAllWindows():
    ''' Destroys all windows '''
    if _SHOW_PREVIEW:
        cv2.destroyAllWindows


def showImage(window, frame):
    ''' Show a frame in a named window '''
    if _SHOW_PREVIEW:
        logger.debug('Show image on window: %s' % window)
        cv2.imshow(window, frame)
        cv2.waitKey(1)
        

logger = config.getLogger()

logger.info("Hello! Getting ready to spot some trash pandas!")

raccoonRecognizer = RacoonRecognizer(type="AZURE")

isRaspberry = platform.uname()[0] == "Linux"
logger.info("Platform detection.. Raspberry Pi? %s" % isRaspberry)

logger.info("Starting video feed...")
#vs = VideoStream(usePiCamera=isRaspberry, resolution=(1024, 768)).start()

vs = VideoStream(usePiCamera=False, src="rtsp://Wyze:Camera@192.168.1.135/live", resolution=(1280, 720)).start()
logger.info("Waiting 2 seconds to start up the camera stream...")
time.sleep(2.0)

motionCounter = 0
lastShown = datetime.datetime.now()
text = ""
avg = None

# Ensure that the results path exists
_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

initializeWindow("preview")

# Variables
logger.info("Starting to watch video stream")
while True:
    try:
        time.sleep(_CAPTURE_FPS / 60)
        timestamp = datetime.datetime.now()

        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the average frame is None, initialize it
        if avg is None:
            logger.info("Starting background model...")
            avg = gray.copy().astype("float")
            continue

        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

        # threshold the delta image, dilate the thresholded image to fill
        # in holes, then find contours on thresholded image
        thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contour_bounding_area = [None, None, None, None]  # (X, Y, X', Y')
        isolation_frame = None

        logger.debug("shape: " + str(frame.shape))

        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < _CONTOUR_TOLERANCE:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the
            (x, y, w, h) = cv2.boundingRect(c)

            logger.debug("contour: %d %d %d %d" % (x, y, w, h))

            if contour_bounding_area[0] is None:
                contour_bounding_area = [x, x + w, y, y + h]
            else:
                contour_bounding_area[0] = min(contour_bounding_area[0], x)
                contour_bounding_area[1] = max(contour_bounding_area[1], x + w)
                contour_bounding_area[2] = min(contour_bounding_area[2], y)
                contour_bounding_area[3] = max(contour_bounding_area[3], y + h)

        # If a change area is found, define a new frame with only the change area
        if contour_bounding_area[0] is not None:
            # Crop the frame to the area where motion occurred
            isolation_frame = frame[contour_bounding_area[2]:contour_bounding_area[3],
                                    contour_bounding_area[0]:contour_bounding_area[1]]

            cv2.rectangle(frame, (contour_bounding_area[0], contour_bounding_area[2]),
                    (contour_bounding_area[1], contour_bounding_area[3]),
                    (0, 255, 0), 2)

            logger.debug(contour_bounding_area)
            logger.debug("isolation_frame shape: " + str(isolation_frame.shape))

        # check to see if the room is occupied
        if isolation_frame is not None:
            # check to see if enough time has passed between uploads
            if (timestamp - lastShown).seconds >= _MOTION_ELAPSED_TIME_TOLERANCE:
                showImage("preview", frame)
                # cv2.imshow("isolation_frame", isolation_frame)
                lastShown = timestamp

                if (_DEBUG):                
                    cv2.imwrite("last_movement.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

                # Trigger the recognizer
                isRaccoon = raccoonRecognizer.hasRaccoon(isolation_frame)
                if isRaccoon is True:
                    filename = str(_RESULTS_PATH / (timestamp.strftime("%Y-%m-%d_%H%M%S") + ".jpg"))
                    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    logger.info(f"Raccoon found! Saving image to '{filename}'")
                else:                    
                    logger.info("False alarm. Something moved but it wasn't a raccoon.")
            
    except KeyboardInterrupt:
        break

logger.info("Cleaning up...")

# Clean up...
destroyAllWindows()
vs.stop()

logger.info("Goodbye!\n")