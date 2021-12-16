import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore') 

import pandas as pd
import torch
import cv2
from IPython.display import Image
from matplotlib import pyplot as plt
import os
import glob
from tqdm import tqdm
from flask_ngrok import run_with_ngrok
from flask import Flask, jsonify, request
import numpy as np
from numpy import load
import json


from azure.storage.blob import BlobServiceClient
# Create the BlobServiceClient object which will be used to create a container client
connect_str = 'DefaultEndpointsProtocol=https;AccountName=fanpics;AccountKey=8XYaGSztvUqMKmsBQ9pG5lVu4Gf/eHvE4YbKvwzTbJORIieevaXzOBL6KZck4PDxD42iRzrbQTbi5B3jfXv0XQ==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

app = Flask(__name__)
#run_with_ngrok(app)

os.chdir('/home/USAI001')
###Set parameter --------
path_yolov5 = 'yolov5'
path_model = 'yolov5/weights_FanDetect/Modelyolo5_FanDetect.pt'
#path_imgs = 'yolov5/test/'
### ----- Load Model -------------
model = torch.hub.load(path_yolov5, 'custom', path=path_model, source='local', device='cpu')  # local repo
### Seting Optional --------------------------------
model.conf = 0.50  # NMS confidence threshold
model.iou = 0.50  # NMS IoU threshold
model.classes = None   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

@app.route("/predictFanDetect",methods=["POST"])
def predict():
    if request.method == "POST":
        input_value = request.form["path_images"]
        
        # Create a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(container='fan', blob=input_value)
        with open(input_value, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        #input_value = request.values["path_images"]
        xmin, ymin, xmax, ymax, Confidence = predict_Fan(input_value)
        print(f"xmin : {xmin}")
        print(f"ymin : {ymin}")
        print(f"xmax : {xmax}")
        print(f"ymax : {ymax}")
        print(f"Confidence : {Confidence}")
        json_data = json.dumps({'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax, 'Confidence':Confidence})
        #results.append(json_data)
    return json_data

def predict_Fan(path_imgs):  ## input format == .jpg or png
    img_c = cv2.imread(path_imgs)
    # Inference
    results = model(path_imgs, size=640)  # custom inference size
    pred = results.pandas().xyxy[0].sort_values('confidence')
    xmin = pred['xmin'][0]/img_c.shape[1]
    ymin = pred['ymin'][0]/img_c.shape[0]
    xmax = pred['xmax'][0]/img_c.shape[1]
    ymax = pred['ymax'][0]/img_c.shape[0]
    Confidence = pred['confidence'][0] 
    
    return xmin, ymin, xmax, ymax, Confidence

#----------- use def  --------------------------- 

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5006)
    #app.run(debug=True, port=5050)
    #app.run(host="0.0.0.0", port=6005 ,debug=False)
#     app.run(host="localhost", port="5050", debug=False)
