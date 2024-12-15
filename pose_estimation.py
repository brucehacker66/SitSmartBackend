import os
import torch
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# Initialize model and device
model = None

def initialize_model():
    global model
    model =  mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)

def inference(images, conf_threshold = 0.5):
    """
    Perform inference on a batch of images using MediaPipe Pose to detect keypoints.

    Args:
        images (list): List of PIL.Image images.

    Returns:
        list: A list of keypoints for each image. Each element is a list of tuples (x, y, z, visibility).
              The order of keypoints matches the order of the input image list.
    """
    global model
    keypoints_list = []  # To store the keypoints for each image in the same order.

    for idx, image in enumerate(images):
        # Convert PIL.Image to numpy array and then to RGB for MediaPipe processing.
        image_np = np.array(image)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Pose.
        results = model.process(image_rgb)
        
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                x = landmark.x if landmark.visibility > conf_threshold else np.nan
                y = landmark.y if landmark.visibility > conf_threshold else np.nan
                z = landmark.z if landmark.visibility > conf_threshold else np.nan
                keypoints.append([x, y,z])
            keypoints_list.append(keypoints)
        else:
            # Append an empty list if no keypoints are detected
            keypoints_list.append(np.full((33, 3), np.nan))
        print(f"Image {idx}: Keypoints - {keypoints_list}")

    return keypoints_list
