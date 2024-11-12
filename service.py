from pose_estimation import initialize_model, inference
import os
from PIL import Image
import fnmatch
from pathlib import Path

def readImages(num_images, localStoragePath, userId="user123"):
  images = []
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
  )[:num_images]  # only take the most recent `num_images` files

  for file_path in files:
      try:
          img = Image.open(file_path)
          images.append(img)
      except Exception as e:
        print(f"Error opening image {file_path}: {e}")

  return images



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
    # # local storage path
    # data_path = r"D:\JHU24-25\MLSys\data"
    # # Get all files in the data_path folder
    # image_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    # images = [Image.open(os.path.join(data_path, image_file)) for image_file in image_files]

    downloads_path = Path.home() / "Downloads"
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