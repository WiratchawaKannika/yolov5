import pandas as pd
import torch
import cv2
import os
import glob

os.chdir('/home/USAI001')
###Set parameter --------
path_yolov5 = 'yolov5'
path_model = 'yolov5/weights_FanDetect/Modelyolo5_FanDetect.pt'
path_imgs = 'yolov5/test/'

def predict_Fan(imgs):  ## input format == .jpg or .png
    ### ----- Load Model -------------
    model = torch.hub.load(path_yolov5, 'custom', path=path_model, source='local', device='cpu')  # local repo

    ### Seting Optional --------------------------------
    model.conf = 0.50  # NMS confidence threshold
    model.iou = 0.50  # NMS IoU threshold
    model.classes = None   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image
    #.cpu()  # CPU

    # Image
    #imgs = path_imgs + 'FanDetect_custom_359.jpg'

    # Inference
    results = model(imgs, size=640)  # custom inference size
    pred = results.pandas().xyxy[0].sort_values('confidence')
    
    xmin = pred['xmin'][0]
    ymin = pred['ymin'][0]
    xmax = pred['xmax'][0]
    ymax = pred['ymax'][0]
    Confidence = pred['confidence'][0] 
    
    return xmin, ymin, xmax, ymax, Confidence


#----------- use def  --------------------------- 
## Image
imgs = path_imgs + 'FanDetect_custom_359.jpg'
xmin, ymin, xmax, ymax, Confidence = predict_Fan(imgs)

print('*'*30)
print('Fan Detection')
print('-'*30)
print(f"xmin : {xmin}\n")
print(f"ymin : {ymin}\n")
print(f"xmax : {xmax}\n")
print(f"ymax : {ymax}\n")
print(f"Confidence : {Confidence}")
print('-'*30)
