import cv2
import os
import config

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from pprint import pprint


class GCPRaccoonRecognizer:
    def __init__(self):
        self._TEMP_FILENAME = "capture.jpg"
        
        self.client = vision.ImageAnnotatorClient()        
        self.logger = config.getLogger()

    def hasRaccoon(self, in_image):
        cv2.imwrite(self._TEMP_FILENAME, in_image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with open(self._TEMP_FILENAME, "rb") as image_stream:
            self.logger.info("Analyze image....")
            content = image_stream.read()
               
        image = types.Image(content=content)           
        response = self.client.label_detection(image=image)        

        foundRaccoon = False
        for label in response.label_annotations:
            self.logger.info(f"Description: {label.description}, Score: {label.score}, Topicality: {label.topicality}")
            if label.description in ('Mammal', 'Procyonidae', 'Procyon') and label.score > 0.6:
                foundRaccoon = True

        if foundRaccoon is True:
            self.logger.info("Found it")

        os.remove(self._TEMP_FILENAME)

        return foundRaccoon
