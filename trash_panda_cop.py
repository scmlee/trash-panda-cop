import cv2
import imutils
import datetime
import time
import azure_raccoon_recognizer  # as raccoon_recognizer
from imutils.video import VideoStream


_DEBUG = False  # Toggle for debug output statements
_CONTOUR_TOLERANCE = 8000
_MOTION_ELAPSED_TIME_TOLERANCE = 3.0
_CAPTURE_FPS = 10


def debug(*args):
    '''
    Debug output; based on global _debug flag
    '''
    if _DEBUG:
        print(*args)


cv2.namedWindow("preview")
cv2.namedWindow("isolation_frame")

# vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FPS, 2)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

motionCounter = 0
lastShown = datetime.datetime.now()
text = ""
avg = None

# Variables
azure_region = 'westcentralus' #Here you enter the region of your subscription
azure_url = 'https://{}.api.cognitive.microsoft.com/vision/v2.0/analyze'.format(azure_region)
azure_key = '034d1f5778624292ad1070373ac71d2a'
azure_maxNumRetries = 10

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    time.sleep(_CAPTURE_FPS / 60)

    timestamp = datetime.datetime.now()
    frame = imutils.resize(frame, width=1024)
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
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        debug("contour:", x, y, w, h)

        if contour_bounding_area[0] is None:
            contour_bounding_area = [x, x + w, y, y + h]
        else:
            contour_bounding_area[0] = min(contour_bounding_area[0], x)
            contour_bounding_area[1] = max(contour_bounding_area[1], x + w)
            contour_bounding_area[2] = min(contour_bounding_area[2], y)
            contour_bounding_area[3] = max(contour_bounding_area[3], y + h)

    # draw the text and timestamp on the frame
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, "{}".format(ts), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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
    cv2.imshow("preview", frame)

    # check to see if the room is occupied
    if isolation_frame is not None:
        # check to see if enough time has passed between uploads
        if (timestamp - lastShown).seconds >= _MOTION_ELAPSED_TIME_TOLERANCE:

            # Trigger the recognizer
            # azure_raccoon_recognizer.hasRaccoon(isolation_frame)

            cv2.imshow("isolation_frame", isolation_frame)
            lastShown = timestamp

    rval, frame = vc.read()

    # Escape to exit
    key = cv2.waitKey(20)
    if key == 27:
        break

# Clean up...
# cv2.destroyWindow("preview")
cv2.destroyAllWindows
vc.release()
