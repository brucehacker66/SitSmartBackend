from ultralytics import YOLO
import os
import torch
from PIL import Image
import numpy as np

# Initialize model and device
model = None
device = None

def initialize_model():
    global model, device
    model_path = os.path.join("checkpoints", "yolov8s-pose.pt")
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

def inference(images, conf_threshold=0.5):
    """
    Perform inference on a list of PIL Images and return keypoints for detected humans.

    Args:
    - images (list): List of PIL Image objects.
    - conf_threshold (float): Confidence threshold for YOLO.

    Returns:
    - results_list (list of list): List containing keypoints for detected humans in each image. Keypoint is in a list [x, y].
    - keypoint of [0, 0] means the keypoint is not detected, like elbow, wrist, etc. They are still appended to the result because their prescence can be signal for bad posture.
        Nose
        Left Eye
        Right Eye
        Left Ear
        Right Ear
        Left Shoulder
        Right Shoulder
        Left Elbow
        Right Elbow
        Left Wrist
        Right Wrist
        Left Hip
        Right Hip
        Left Knee
        Right Knee
        Left Ankle
        Right Ankle
    """
    results_list = []

    for image in images:
        # Convert PIL Image to the format YOLO expects (numpy array)
        image_np = np.array(image)

        # Run inference on the image
        results = model(source=image_np, conf=conf_threshold, show=False, device=device)
        # Extract keypoints from results if human detected
        for result in results:
            if hasattr(result, "names"):
                if result.names[0] == "person":
                    if hasattr(result, "boxes"):
                        if hasattr(result, "keypoints"):
                            keypoints = result.keypoints.xy.cpu().tolist()  # Get keypoints as a list
                            classes = result.boxes.cls.cpu().numpy()
                            for cls_idx, kp in zip(classes, keypoints):
                                if cls_idx == 0:
                                    results_list.append(kp)

    return results_list

