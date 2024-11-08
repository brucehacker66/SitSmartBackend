from ultralytics import YOLO
import os
import torch

model_path = os.path.join("checkpoints", "yolov8n-pose.pt")
model = YOLO(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def inference(image_path, save_path="results"):
    results = model(source=image_path, conf=0.9, show=True)
    
    return results

# test 
data_path = r"D:\JHU24-25\MLSys\data"
image_path = os.path.join(data_path, "0_Color_1729735655498.42675781250000.png")

inferece(image_path)