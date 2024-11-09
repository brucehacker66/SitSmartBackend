from pose_estimation import initialize_model, inference
import os
from PIL import Image

def readImages(interval, localStoragePath, userId):
  pass

def postureDetect(interval=60):
  """
    Posture detection pipeline. This function reads images from the local storage, performs inference on them, and sends the results to the server.

    Args:
    - interval (int): Interval in seconds to read images from the local storage.

    """
  initialize_model()
  results = inference(images)
  return results


# testing
if __name__ == "__main__":
    # local storage path
    data_path = r"D:\JHU24-25\MLSys\data"
    # Get all files in the data_path folder
    image_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    images = [Image.open(os.path.join(data_path, image_file)) for image_file in image_files]

    results = postureDetect(images)

    assert len(results) == len(images)

    # Print keypoints results
    for idx, keypoints in enumerate(results):
        print(f"Image {idx}: Keypoints - {keypoints}")