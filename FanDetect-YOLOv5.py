import pandas as pd
import torch
import cv2
import os
import glob

os.chdir('/home/usaikku/api-FanDetect-YOLOv5')
###Set parameter --------
path_yolov5 = 'yolov5'
path_model = 'weights/Modelyolo5_FanDetect.pt'
path_imgs = 'test/'

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
imgs = path_imgs + 'FanDetect_custom_359.jpg'

# Inference
results = model(imgs, size=640)  # custom inference size
pred = results.pandas().xyxy[0].sort_values('confidence')

print('*'*50)
print('Fan Detection')
print('-'*50)
print(f"xmin : {pred['xmin'][0]}\n")
print(f"ymin : {pred['ymin'][0]}\n")
print(f"xmax : {pred['xmax'][0]}\n")
print(f"ymax : {pred['ymax'][0]}\n")
print(f"Confidence : {pred['confidence'][0]}")
print('-'*50)
