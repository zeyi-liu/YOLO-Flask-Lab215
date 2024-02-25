# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Perform test request
"""

import pprint

import requests

DETECTION_URL = 'http://127.0.0.1:5000'
IMAGE = r"E:\Desktop\CVproj\DefectDetection\code\YOLOV5\CSU-DET\train\images\Basler_a2A5320-7gcBAS__40073191__20240116_114624773_0005_1600_4160.jpg"

# Read image
with open(IMAGE, 'rb') as f:
    image_data = f.read()

response = requests.post(DETECTION_URL, files={'file': image_data}).json()

pprint.pprint(response)
