"""
Raccoon Recognizer module (Azure-based)
"""

import cv2
import os
import config
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

_KEY = os.getenv("AZURE_API_KEY") 
_TEMP_FILENAME = "capture.jpg"

credentials = CognitiveServicesCredentials(_KEY)       
client = ComputerVisionClient('https://canadacentral.api.cognitive.microsoft.com/', credentials)
logger = config.getLogger()


def hasRaccoon(in_image):
    cv2.imwrite(_TEMP_FILENAME, in_image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    with open(_TEMP_FILENAME, "rb") as image_stream:
        logger.info("Analyze image....")
        image_analysis = client.analyze_image_in_stream(image=image_stream,
                                                        visual_features=[VisualFeatureTypes.tags]);

    foundRaccoon = False
    for tag in image_analysis.tags:
        logger.info(tag)
        if tag.name in ('animal', 'raccoon') and tag.confidence > 0.6:
            foundRaccoon = True

    if foundRaccoon is True:
        logger.info("Found it")

    os.remove(_TEMP_FILENAME)

    return foundRaccoon
