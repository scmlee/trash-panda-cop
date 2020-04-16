from logzero import logger, setup_logger, logging
import config

class RacoonRecognizer:
    def __init__(self, type = "AZURE"):
        logger = config.getLogger()

        # Initialize either Azure of Google version of the recognizer        
        if (type == "AZURE"):
            logger.info("Intialize Azure version...")   
            from .azure_raccon_recognizer import AzureRaccoonRecognizer
            self.recognizer = AzureRaccoonRecognizer()
        elif (type == "GCP"):
            logger.info("Intialize GCP version...")   
            from .gcp_raccoon_recognizer import GCPRaccoonRecognizer
            self.recognizer = GCPRaccoonRecognizer()
        else:
            logger.info("Invalid type...")   

    def hasRaccoon(self, in_image):
        return self.recognizer.hasRaccoon(in_image)
