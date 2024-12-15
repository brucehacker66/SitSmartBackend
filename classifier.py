from pose_estimation import initialize_model, inference
import os
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pickle


def readImages(interval, localStoragePath, userId):
  pass

def postureDetect(interval=60):
  """
    Posture detection pipeline. This function reads images from the local storage, performs inference on them, and sends the results to the server.

    Args:
    - interval (int): Interval in seconds to read images from the local storage.

    """
#   initialize_model()
  results = inference(images)
  return results


# class PostureFeatureExtractor:
def extract_features(keypoints):
    """
    Extract features from 3D keypoints, adding a vertical nose offset feature
    to help distinguish backward vs. normal posture.

    Args:
        keypoints (np.array): shape (17, 3), each row is (x, y, z)

    Returns:
        np.array: Extracted features (including vertical_nose_offset)
    """
    # Indices (assuming COCO-style or similar)
    nose = keypoints[0]            # (x, y, z)
    left_shoulder = keypoints[11]  # (x, y, z)
    right_shoulder = keypoints[12] # (x, y, z)
    if np.isnan(nose).any():
        return np.full((8,), np.nan)

    # 1. Shoulder line & length
    shoulder_vector = right_shoulder - left_shoulder
    shoulder_length = np.linalg.norm(shoulder_vector)

    # Horizontal angle (projection onto the XY plane)
    horizontal_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])

    # Vertical angle (projection onto the XZ plane)
    vertical_angle = np.arctan2(
        shoulder_vector[2],
        np.sqrt(shoulder_vector[0]**2 + shoulder_vector[1]**2)
    )

    # Depth angle (projection onto the YZ plane)
    depth_angle = np.arctan2(
        shoulder_vector[2],
        np.sqrt(shoulder_vector[0]**2 + shoulder_vector[1]**2 + shoulder_vector[2]**2)
    )

    # 3. Nose deviation from shoulder line (existing feature)
    if shoulder_length != 0:
        shoulder_unit_vector = shoulder_vector / shoulder_length
    else:
        shoulder_unit_vector = np.array([0, 0, 0])  # degenerate case

    nose_projection = np.dot(nose - left_shoulder, shoulder_unit_vector)
    nose_deviation = np.linalg.norm((nose - left_shoulder) - nose_projection * shoulder_unit_vector)

    # 4. Z offset (if you already have it)
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    z_offset = nose[2] - mid_shoulder[2]  # forward/back offset

    # 5. **NEW** vertical offset (assuming y is vertical)

    # Existing distance calculation
    nose_distance = np.linalg.norm(nose - mid_shoulder)

    nose_angle = calculate_angle_nose_shoulder_yaxis(nose, left_shoulder, right_shoulder)

#         # Calculate shoulder width to use as a normalizing factor
#         shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

#         # Normalized nose distance
#         normalized_nose_distance = nose_distance / shoulder_width if shoulder_width != 0 else 0

    # Combine into a bigger feature vector
    features = np.array([
        shoulder_length,
        horizontal_angle,
        vertical_angle,
        depth_angle,
        nose_deviation,
        z_offset,             # existing feature for forward/back separation
        nose_distance,  # new feature for backward/normal separation
        nose_angle
    ])

    return features


def calculate_angle_nose_shoulder_yaxis(nose_vector, left_shoulder, right_shoulder):
    """
    Calculate the angle between the nose position and the shoulder midpoint relative to the y-axis.

    Parameters:
        nose_vector (numpy array): A vector [x, y, z] representing the nose position.
        left_shoulder (numpy array): A vector [x, y, z] representing the left shoulder position.
        right_shoulder (numpy array): A vector [x, y, z] representing the right shoulder position.

    Returns:
        float: The angle in radians between the nose and the shoulder midpoint relative to the y-axis.
    """
    # Calculate the midpoint of the shoulders
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0

    # Calculate the vector difference between the nose and shoulder midpoint
    vector_diff = nose_vector - shoulder_midpoint

    # Define the y-axis vector
    y_axis_vector = np.array([0, 1, 0])

    # Calculate the magnitudes of the vectors
    mag_vector_diff = np.linalg.norm(vector_diff)
    mag_y_axis = np.linalg.norm(y_axis_vector)

    # Prevent division by zero
    if mag_vector_diff == 0:
        raise ValueError("Vector difference is zero, cannot compute angle.")

    # Calculate the dot product
    dot_product = np.dot(vector_diff, y_axis_vector)

    # Calculate the angle (in radians)
    angle = np.arccos(dot_product / (mag_vector_diff * mag_y_axis))

    return angle

def prepare_dataset(results_normal, results_backwards, results_forwards):
    """
    Prepare dataset from different posture types
    Ignore samples where nose is NaN
    
    Args:
        results_normal (np.array): Good posture keypoints
        results_backwards (np.array): Bad posture - backwards
        results_forwards (np.array): Bad posture - forwards
    
    Returns:
        tuple: X (features), y (labels)
    """
#     extractor = PostureFeatureExtractor()
    
    # Filter out samples where nose is NaN
    normal_valid_indices = ~np.isnan(results_normal[:, 0]).any(axis=1)
    backwards_valid_indices = ~np.isnan(results_backwards[:, 0]).any(axis=1)
    forwards_valid_indices = ~np.isnan(results_forwards[:, 0]).any(axis=1)
    
    # Filter datasets
    results_normal_filtered = results_normal[normal_valid_indices]
    results_backwards_filtered = results_backwards[backwards_valid_indices]
    results_forwards_filtered = results_forwards[forwards_valid_indices]
    
    # Print filtering information
    print(f"Normal samples: Total {len(results_normal)}, After filtering {len(results_normal_filtered)}")
    print(f"Backwards samples: Total {len(results_backwards)}, After filtering {len(results_backwards_filtered)}")
    print(f"Forwards samples: Total {len(results_forwards)}, After filtering {len(results_forwards_filtered)}")
    
    # Extract features for filtered datasets
    X_normal = np.array([extract_features(sample) for sample in results_normal_filtered])
    X_backwards = np.array([extract_features(sample) for sample in results_backwards_filtered])
    X_forwards = np.array([extract_features(sample) for sample in results_forwards_filtered])
    
    # Create labels
    y_normal = np.zeros(len(X_normal))  # 0 for normal posture
    y_backwards = np.ones(len(X_backwards))  # 1 for backwards posture
    y_forwards = np.full(len(X_forwards), 2)  # 2 for forwards posture
    
    # Combine datasets
    X = np.vstack((X_normal, X_backwards, X_forwards))
    y = np.concatenate((y_normal, y_backwards, y_forwards))
    
    return X, y



def train_and_evaluate_model(X, y):
    """
    Train and evaluate classification model
    
    Args:
        X (np.array): Features
        y (np.array): Labels
    
    Returns:
        tuple: Trained model, classification report
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM Classifier
    classifier = SVC(kernel='rbf', random_state=42)
    classifier.fit(X_train_scaled, y_train)
    joblib.dump(classifier, './model_checkpoint/svm_latest.joblib')

    # Saving the scaler object to disk
    with open('./model_checkpoint/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    
    # Predict and evaluate
    y_pred = classifier.predict(X_test_scaled)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'Backwards', 'Forwards'])
    
    # Generate confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    
    # # Visualize confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', 
    #             xticklabels=['Normal', 'Backwards', 'Forwards'],
    #             yticklabels=['Normal', 'Backwards', 'Forwards'])
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.tight_layout()
    # plt.show()
    
    return classifier, report


def classify_posture(images):
    """
    Classify posture based on keypoints.

    Args:
    - images (list): .jpg to be classified

    Returns:
    - prediction (int): Predicted class label or None if classification is not possible.
    """
    with open('./model_checkpoint/scaler.pkl', 'rb') as file:
        loaded_scaler = pickle.load(file)

    keypoints = np.array((postureDetect(images))[0])
    
    # features = extract_features(keypoints)
    features = extract_features(keypoints)
    features_scaled  = loaded_scaler.transform(features.reshape(1, -1))
    if features is not None:
        prediction = classifier.predict(features_scaled)[0]
        return prediction
    else:
        return None


# testing
if __name__ == "__main__":
    # Load keypoints data
    results_normal = np.load('./keypoints/normal_keypoints/keypoints_img_coord.npy')
    results_backwards = np.load('./keypoints/backward_keypoints/keypoints_img_coord.npy')
    results_forwards = np.load('./keypoints/forward_keypoints/keypoints_img_coord.npy')
    
    # Prepare dataset
    X, y = prepare_dataset(results_normal, results_backwards, results_forwards)
    
    # Train and evaluate
    classifier, report = train_and_evaluate_model(X, y)
    
    # Print classification report
    print(report)

    joblib.dump(classifier, './model_checkpoint/svm_latest.joblib')

    # Loading the scaler object from disk


    image_path = '../AI System Data/Backwards/frames/frame_00028.jpg'  # Example using the first image from normal posture
    images = [Image.open(image_path)]
    initialize_model()
    prediction = classify_posture(images)
    posture_classes = {0: 'Good Posture', 1: 'Bad Posture - Backwards', 2: 'Bad Posture - Forwards'}
    if prediction is not None:
        print(f"Predicted Posture: {posture_classes[prediction]}")
    else:
        print("Could not classify posture due to missing keypoints.")



    # # DO INFERENCE HERE
    # image_path = '../AI System Data/Normal posture/frames/frame_00008.jpg'  # Example using the first image from normal posture
    # images = [Image.open(image_path)]

    # prediction = classify_posture(images)
    # posture_classes = {0: 'Good Posture', 1: 'Bad Posture - Backwards', 2: 'Bad Posture - Forwards'}
    # if prediction is not None:
    #     print(f"Predicted Posture: {posture_classes[prediction]}")
    # else:
    #     print("Could not classify posture due to missing keypoints.")

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
