import os
from pathlib import Path
from PIL import Image
from threading import Thread, Event
import time
import fnmatch
from pose_estimation import initialize_model, inference
import joblib
from classifier import extract_features
import numpy as np

# Get the default Downloads folder, regardless of the OS
def get_default_download_folder():
    return str(Path.home() / "Downloads")

user_status = {}          # Maps user_id to their current status
user_intervals = {}       # Maps user_id to their update interval
default_interval = 45     # Default interval in seconds
threads = {}              # Maps user_id to their respective thread
stop_events = {}          # Maps user_id to an Event to stop the thread if needed

# Initialize model for inference
initialize_model()

def read_images(num_images, userId="user123"):
    images = []
    localStoragePath = get_default_download_folder()  # Use default Downloads folder
    naming_scheme = f"sitsmart_{userId}_*.png"
    # Filter files by userId and naming scheme
    files = [
        f for f in os.listdir(localStoragePath)
        if fnmatch.fnmatch(f, naming_scheme) and f.endswith('png')
    ]
    files = sorted(
        [os.path.join(localStoragePath, f) for f in files],
        key=os.path.getmtime,
        reverse=True
    )[:num_images]  # Only take the most recent `num_images` files

    for file_path in files:
        try:
            img = Image.open(file_path)
            images.append(img)
        except Exception as e:
            print(f"Error opening image {file_path}: {e}")

    return images

def classify_posture(keypoints, classifier):
    """
    Classify posture based on keypoints.

    Args:
    - keypoints (list): list of (n,17,2) keypoints from yolo, where n is the number of images

    Returns:
    - prediction (list): list of predicted class label, each element corresponds to the class of the image at that index; nan if can't be classified.
    """
    prediction = []
    print(keypoints)
    for i in np.arange(len(keypoints)):
        features = extract_features(keypoints[i])
        if features is not None:
            prediction.append(classifier.predict([features])[0])
            # return prediction
        else:
            prediction.append(np.nan)
    return prediction


def posture_detect(user_id, capture_interval = 5, detection_interval = 45):
    """
    Main Posture detection pipeline for a specific user.
    This function reads images, performs inference, and updates user status.
    """
    images = read_images(num_images=detection_interval // capture_interval, userId=user_id)
    classifier = joblib.load('./model_checkpoint/gradient_boosting_model.joblib')

    
    if images:
        # perform yolo inference 
        keypoints = np.array(inference(images))
        # perform classification
        prediction = classify_posture(keypoints, classifier)

        count_bad = sum(1 for x in prediction if x == 1 or x == 2)
        count_good = sum(0 for x in prediction if x == 0)
        num_img = detection_interval // capture_interval

        # Logic to update status based on prediction results
        if count_good >=  num_img//2 + 1:
            user_status[user_id] = "Good"
        elif count_bad >= 0.7 * num_img:
            user_status[user_id] = "Bad"
        else:
            user_status[user_id] = "Unknown"

def update_status_periodically(user_id):
    """
    Periodically update the posture status of a user.
    """
    while not stop_events[user_id].is_set():
        interval = user_intervals.get(user_id, default_interval)
        posture_detect(user_id, detection_interval=interval)
        time.sleep(interval)

def add_user(user_id = "user123"):
    """
    Add a new user and start their status update thread.
    """
    if user_id in user_status:
        raise ValueError("User already exists")

    # Initialize user status and interval
    user_status[user_id] = "Unknown"
    user_intervals[user_id] = default_interval

    # Create an event to stop the thread if needed
    stop_events[user_id] = Event()

    # Start a new thread for the user
    thread = Thread(target=update_status_periodically, args=(user_id,), daemon=True)
    threads[user_id] = thread
    thread.start()

def set_user_interval(user_id, interval):
    """
    Set the update interval for a specific user.
    """
    if user_id not in user_intervals:
        raise ValueError("User does not exist")

    user_intervals[user_id] = interval

def get_posture_status(user_id):
    """
    Get the current posture status of a user.
    """
    return user_status.get(user_id, "Processing")

def user_exists(user_id):
    """
    Check if a user exists.
    """
    return user_id in user_status


# testing
if __name__ == "__main__":
    # # local storage path
    # data_path = r"D:\JHU24-25\MLSys\data"
    # # Get all files in the data_path folder
    # image_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    # images = [Image.open(os.path.join(data_path, image_file)) for image_file in image_files]

    downloads_path = Path(os.path.expanduser("~/Downloads"))
    if not downloads_path.exists():
        print("Downloads folder not found.")
    
    # 5 sec intervals, 45s window -> 9 images
    images = readImages(9, downloads_path)

    # print(f"Number of images read: {len(images)}")
    # for i, img in enumerate(images):
    #     print(f"Image {i+1}: {img}")

    results = postureDetect(images)

    assert len(results) == len(images)

    # Print keypoints results
    for idx, keypoints in enumerate(results):
        print(f"Image {idx}: Keypoints - {keypoints}")