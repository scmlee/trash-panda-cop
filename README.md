# Trash Panda Cop
RaspberryPi based project to detect and deter raccoon, aka Trash Panda, visits to my backyard!

## Detection
The use of AI to detect the presence of Trash Pandas!

## Deterrence
Once detected, triggering of various deterrents to communicate sense of unwelcomeness to our furry fiends!

## Usage
1. Run `pip install -r requirements.txt` to install required Python modules
2. Set the respective cloud providers' API keys for image recognition service.

    To use __Azure Cognitive Services__, obtain an API Key for Computer Vision services and set `AZURE_API_KEY` environment variable with API Key

    To use __Google Cloud Platform's Cloud Vision services__, create a credentials file and set `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to the file
3. Run `python trash_panda_cop.py`

## Inspirations
[Motion-Activated Water Gun Turret (YouTube)] (https://www.youtube.com/watch?v=Jmy2lWDBTf8&t=76s)

## Dependencies

### Linux:
* livffi-dev (required for Python cryptography package)
* libssl-dev
* libcblas-dev 
* libhdf5-dev
* libhdf5-serial-dev
* libatlas-base-dev
* libjasper-dev
* libqtgui4
* libqt4-test

### Python:
* opencv-python
* imutils
* PiCamera
* azure-cognitiveservices-vision-computervision (azure-sdk-for-python)
* google-cloud-vision

## References
