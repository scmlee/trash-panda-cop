"""
Raccoon Recognizer module (Azure-based)
"""
import cv2
import os
import config
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

class AzureRaccoonRecognizer:

    def __init__(self):
        self._KEY = os.getenv("AZURE_API_KEY") 
        self._TEMP_FILENAME = "capture.jpg"

        credentials = CognitiveServicesCredentials(self._KEY)       
        self.client = ComputerVisionClient('https://canadacentral.api.cognitive.microsoft.com/', credentials)
        self.logger = config.getLogger()

    def hasRaccoon(self, in_image):
        cv2.imwrite(self._TEMP_FILENAME, in_image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with open(self._TEMP_FILENAME, "rb") as image_stream:
            self.logger.info("Analyze image....")
            image_analysis = self.client.analyze_image_in_stream(image=image_stream,
                                                            visual_features=[VisualFeatureTypes.tags]);

        foundRaccoon = False
        for tag in image_analysis.tags:
            self.logger.info(tag)
            if tag.name in ('animal', 'raccoon') and tag.confidence > 0.6:
                foundRaccoon = True

        if foundRaccoon is True:
            self.logger.info("Found it")

        os.remove(self._TEMP_FILENAME)

        return foundRaccoon
