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

def inference(images, conf_threshold=0.5):
    """
    Perform inference on a list of PIL Images and return keypoints for detected humans.

    Args:
    - images (list): List of PIL Image objects.
    - conf_threshold (float): Confidence threshold for Mediapipe.

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
        # Convert PIL Image to numpy array and prepare for MediaPipe
        image_np = np.array(image)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to RGB

        # Run inference using MediaPipe Pose
        with mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=conf_threshold, model_complexity=2
        ) as pose:
            results = pose.process(image_rgb)

            # Extract keypoints if detections are found
            if results.pose_landmarks:
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    x = landmark.x * image_np.shape[1] if landmark.visibility > conf_threshold else 0
                    y = landmark.y * image_np.shape[0] if landmark.visibility > conf_threshold else 0
                    keypoints.append([x, y])

                results_list.append(keypoints)
            else:
                # Append an empty list if no keypoints are detected
                results_list.append([])

    return results_list

