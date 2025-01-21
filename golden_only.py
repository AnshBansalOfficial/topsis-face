import cv2
import mediapipe as mp
import math

# Function to calculate Euclidean distance between two landmarks
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Function to calculate the Golden Ratio score
def calculate_golden_ratio(image_path):
    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return None

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe Face Mesh
    result = face_mesh.process(rgb_image)
    
    if not result.multi_face_landmarks:
        print("No face detected!")
        return None

    landmarks = result.multi_face_landmarks[0].landmark

    # Extract key points
    key_points = {
        "forehead_to_chin": (landmarks[10], landmarks[152]),  # Top of forehead to chin
        "jaw_width": (landmarks[234], landmarks[454]),       # Jawline width
        "eye_width": (landmarks[33], landmarks[263]),        # Distance between eyes
        "nose_width": (landmarks[51], landmarks[281]),       # Nose width
        "eye_to_lips": (landmarks[13], landmarks[14]),       # Eye center to lips center
    }

    # Calculate proportions
    proportions = {
        "face_length_to_width": calculate_distance(key_points["forehead_to_chin"][0], key_points["forehead_to_chin"][1]) /
                                calculate_distance(key_points["jaw_width"][0], key_points["jaw_width"][1]),
        "eye_width_to_nose_width": calculate_distance(key_points["eye_width"][0], key_points["eye_width"][1]) /
                                   calculate_distance(key_points["nose_width"][0], key_points["nose_width"][1]),
        "eye_to_lips_ratio": calculate_distance(key_points["eye_to_lips"][0], key_points["eye_to_lips"][1]) /
                             calculate_distance(key_points["nose_width"][0], key_points["nose_width"][1]),
    }

    # Calculate deviations from the Golden Ratio
    phi = 1.618  # Golden Ratio
    deviations = [(proportion / phi) for proportion in proportions.values()]

    # Calculate the normalized Golden Ratio score
    golden_ratio_score = sum(deviations) / len(deviations)
    normalized_score = min(max(golden_ratio_score, 0), 1)  # Normalize between 0 and 1

    return normalized_score

# Run the calculation on the uploaded image
image_path = "C:/Users/anshi/Desktop/face-topsis/face2.jpg"  # Path to the uploaded image
score = calculate_golden_ratio(image_path)
if score is not None:
    print(f"Golden Ratio Score: {score:.2f}")
