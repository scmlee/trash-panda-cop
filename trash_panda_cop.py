import cv2
import imutils
import datetime
import time
import azure_raccoon_recognizer  # as raccoon_recognizer
from videostream.videostream import VideoStream

_DEBUG = False  # Toggle for debug output statements
_CONTOUR_TOLERANCE = 8000
_MOTION_ELAPSED_TIME_TOLERANCE = 3.0
_CAPTURE_FPS = 10
_SHOW_PREVIEW = False


def log(*args):
    ''' Log output '''
    print(*args)

def debug(*args):
    ''' Debug output; based on global _debug flag '''
    if _DEBUG:
        print(*args)


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
        cv2.imshow(window, frame)


initializeWindow("preview", "isolation_frame")

vs = VideoStream(usePiCamera=False, resolution=(1024, 768)).start()
log("Waiting 2 seconds to start up the camera stream...")
time.sleep(2.0)

motionCounter = 0
lastShown = datetime.datetime.now()
text = ""
avg = None

# Variables
azure_region = 'westcentralus' #Here you enter the region of your subscription
azure_url = 'https://{}.api.cognitive.microsoft.com/vision/v2.0/analyze'.format(azure_region)
azure_key = '034d1f5778624292ad1070373ac71d2a'
azure_maxNumRetries = 10

log("Starting to watch video stream")
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
            print("[INFO] Starting background model...")
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
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_bounding_area = [None, None, None, None]  # (X, Y, X', Y')
        isolation_frame = None

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < _CONTOUR_TOLERANCE:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the
            (x, y, w, h) = cv2.boundingRect(c)
            
            debug("contour:", x, y, w, h)

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

            debug(contour_bounding_area)
            debug("isolation_frame shape: ", isolation_frame.shape)
            # cv2.imshow("isolation_frame", isolation_frame)

        debug("shape: ", frame.shape)
        showImage("preview", frame)
        # cv2.imshow("preview", frame)

        # check to see if the room is occupied
        if isolation_frame is not None:
            # check to see if enough time has passed between uploads
            if (timestamp - lastShown).seconds >= _MOTION_ELAPSED_TIME_TOLERANCE:

                # Trigger the recognizer
                azure_raccoon_recognizer.hasRaccoon(isolation_frame)

                showImage("isolation_frame", isolation_frame)
                # cv2.imshow("isolation_frame", isolation_frame)
                lastShown = timestamp

    # Escape to exit
    # if _SHOW_PREVIEW and cv2.waitKey(20) == 27:
        # break
    # else:    
        # try:
            # time.sleep(0.1)
    except KeyboardInterrupt:
        break

log("Cleaning up...")

# Clean up...
destroyAllWindows()
vs.stop()
