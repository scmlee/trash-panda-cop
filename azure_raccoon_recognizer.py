"""
Raccoon Recognizer module (Azure-based)
"""

import cv2
import os
# import numpy as np
# import tempfile
# import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

_KEY = os.getenv("AZURE_API_KEY") 
_TEMP_FILENAME = "capture.jpg"

credentials = CognitiveServicesCredentials(_KEY)       
client = ComputerVisionClient('https://canadacentral.api.cognitive.microsoft.com/', credentials)


def hasRaccoon(in_image):
    cv2.imwrite(_TEMP_FILENAME, in_image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    with open(_TEMP_FILENAME, "rb") as image_stream:
        print("Analyze image....")
        image_analysis = client.analyze_image_in_stream(image=image_stream,
                                                       visual_features=[VisualFeatureTypes.tags]);

    foundRaccoon = False
    for tag in image_analysis.tags:
        print(tag)
    
    print()
