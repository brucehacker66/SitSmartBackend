from pose_estimation import initialize_model, inference
import os
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier


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



def extract_features(keypoints):
    """
    Extract invariant features from keypoints.

    Args:
    - keypoints (numpy array): Array of keypoints for a single image/frame, shape (17, 2).

    Returns:
    - features (numpy array): Extracted features, shape (2,).
    """
    # Keypoint indices
    # print(np.shape(keypoints))
    nose_idx = 0
    left_shoulder_idx = 5
    right_shoulder_idx = 6

    # Extract keypoints
    nose = keypoints[nose_idx]  # Shape (2,)
    left_shoulder = keypoints[left_shoulder_idx]
    right_shoulder = keypoints[right_shoulder_idx]

    # Check for missing keypoints (coordinates [0, 0] or invalid values)
    if np.all(nose == 0) or np.all(left_shoulder == 0) or np.all(right_shoulder == 0):
        return None
    if np.isnan(nose).any() or np.isnan(left_shoulder).any() or np.isnan(right_shoulder).any():
        return None

    # Compute the midpoint between shoulders
    mid_shoulder = (left_shoulder + right_shoulder) / 2

    # Shift origin to mid_shoulder
    nose_rel = nose - mid_shoulder
    left_shoulder_rel = left_shoulder - mid_shoulder
    right_shoulder_rel = right_shoulder - mid_shoulder

    # Compute shoulder width
    shoulder_width = np.linalg.norm(left_shoulder_rel - right_shoulder_rel)
    if shoulder_width == 0:
        return None

    # Compute normalized nose distance from shoulder midpoint
    nose_distance = np.linalg.norm(nose_rel) / shoulder_width

    # Compute angle between shoulder line and vector to nose
    shoulder_line = right_shoulder_rel - left_shoulder_rel
    nose_vector = nose_rel

    # Compute angle using arccos of the dot product
    dot_product = np.dot(shoulder_line, nose_vector)
    norms_product = np.linalg.norm(shoulder_line) * np.linalg.norm(nose_vector)
    if norms_product == 0:
        return None
    cos_theta = dot_product / norms_product
    # Ensure cos_theta is in the valid range [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)  # In radians

    # Feature vector
    features = [nose_distance, angle]
    return np.array(features)


def process_keypoints_array(keypoints_array, label):
    """
    Process a keypoints array and append features and labels to X and y.

    Args:
    - keypoints_array (numpy array): Shape (N_samples, 17, 2)
    - label (int): Label for the class
    """
    for keypoints in keypoints_array:
        features = extract_features(keypoints)
        if features is not None:
            X.append(features)
            y.append(label)


def classify_posture(images):
    """
    Classify posture based on keypoints.

    Args:
    - images (list): .jpg to be classified

    Returns:
    - prediction (int): Predicted class label or None if classification is not possible.
    """
    keypoints = np.array((postureDetect(images))[0])
    
    features = extract_features(keypoints)
    if features is not None:
        prediction = classifier.predict([features])[0]
        return prediction
    else:
        return None


# testing
if __name__ == "__main__":
    # Load keypoints data
    results_normal = np.load('./npy_data/results_normal.npy')  # Good posture
    results_backwards = np.load('./npy_data/results_backwards.npy')  # Bad posture - backwards
    results_forwards = np.load('./npy_data/results_forwards.npy')  # Bad posture - forwards
    # results_normal = results_normal[normal_i]
    # results_backwards= results_backwards[backwards_i]
    # results_forwards = results_forwards[forward_i]
 
    # Prepare data
    X = []
    y = []


    # Process each class
    process_keypoints_array(results_normal, label=0)  # Good posture
    process_keypoints_array(results_backwards, label=1)  # Bad posture - backwards
    process_keypoints_array(results_forwards, label=2)  # Bad posture - forwards

    X = np.array(X)  # Shape (total_samples, 2)
    y = np.array(y)  # Shape (total_samples,)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the classifier
    #classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                            max_depth=1, random_state=0).fit(X_train, y_train)

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))



    # DO INFERENCE HERE
    image_path = '../AI System Data/Normal posture/frames/frame_00008.jpg'  # Example using the first image from normal posture
    images = [Image.open(image_path)]

    prediction = classify_posture(images)
    posture_classes = {0: 'Good Posture', 1: 'Bad Posture - Backwards', 2: 'Bad Posture - Forwards'}
    if prediction is not None:
        print(f"Predicted Posture: {posture_classes[prediction]}")
    else:
        print("Could not classify posture due to missing keypoints.")

    # import matplotlib.pyplot as plt

    # # Plot features
    # plt.figure(figsize=(8, 6))
    # colors = ['green', 'red', 'blue']
    # labels = ['Good Posture', 'Bad Posture - Backwards', 'Bad Posture - Forwards']

    # for i in range(3):
    #     idx = y == i
    #     plt.scatter(X[idx, 0], X[idx, 1], c=colors[i], label=labels[i], alpha=0.5)

    # plt.xlabel('Normalized Nose Distance')
    # plt.ylabel('Angle (Radians)')
    # plt.legend()
    # plt.title('Feature Distribution by Class')
    # plt.show()
