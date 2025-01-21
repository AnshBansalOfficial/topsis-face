import cv2
import mediapipe as mp
import math
import numpy as np

# Function to calculate Euclidean distance with enhanced precision
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

# Enhanced function to calculate facial symmetry with dynamic normalization and weighted landmarks
def calculate_symmetry_score(landmarks, image_width, image_height):
    # Define left and right landmark pairs for symmetry analysis with weights
    pairs = [
        (33, 263),  # Outer corners of eyes
        (159, 386), # Inner corners of eyes
        (234, 454), # Cheekbones
        (61, 291),  # Mouth corners
        (78, 308),  # Outer edges of lips
        (10, 152)   # Top forehead to chin (vertical symmetry check)
    ]
    
    # Add weights for more important landmarks (e.g., eyes, mouth)
    weights = {
        (33, 263): 2,  # Eyes
        (159, 386): 2,  # Eyes
        (234, 454): 1,  # Cheekbones
        (61, 291): 3,  # Mouth
        (78, 308): 2,  # Lips
        (10, 152): 1,  # Forehead to chin
    }

    # List to store all the symmetry differences calculated
    symmetry_differences = []

    for left_idx, right_idx in pairs:
        left = landmarks[left_idx]
        right = landmarks[right_idx]

        # Convert normalized landmarks to pixel coordinates
        left_x, left_y = int(left.x * image_width), int(left.y * image_height)
        right_x, right_y = int(right.x * image_width), int(right.y * image_height)

        # Horizontal symmetry: Difference in x-coordinates
        horizontal_difference = abs(left_x - (image_width - right_x))

        # Vertical symmetry: Difference in y-coordinates
        vertical_difference = abs(left_y - right_y)

        # Combining horizontal and vertical differences
        weighted_diff = math.sqrt(horizontal_difference**2 + vertical_difference**2) * weights.get((left_idx, right_idx), 1)
        symmetry_differences.append(weighted_diff)

    # Calculate the average of all the differences
    average_symmetry_difference = np.mean(symmetry_differences)

    # Normalize symmetry score (lower difference = higher symmetry)
    max_possible_difference = np.linalg.norm([image_width, image_height])
    normalized_score = 1 - (average_symmetry_difference / max_possible_difference)

    return max(normalized_score, 0)  # Ensure score is non-negative

# Function to calculate Golden Ratio score (optional)
def calculate_golden_ratio_score(landmarks, image_width, image_height):
    # Golden ratio approximation
    golden_ratio = 1.618

    # Key distances for facial proportions (e.g., forehead to chin, eye width, etc.)
    eye_distance = calculate_distance(landmarks[33], landmarks[263])
    mouth_distance = calculate_distance(landmarks[61], landmarks[291])
    forehead_chin_distance = calculate_distance(landmarks[10], landmarks[152])

    # Proportions based on golden ratio
    eye_mouth_ratio = eye_distance / mouth_distance
    mouth_forehead_ratio = mouth_distance / forehead_chin_distance

    # Calculate how close the proportions are to the golden ratio
    eye_mouth_score = abs(eye_mouth_ratio - golden_ratio)
    mouth_forehead_score = abs(mouth_forehead_ratio - golden_ratio)

    golden_ratio_score = (eye_mouth_score + mouth_forehead_score) / 2
    return 1 - golden_ratio_score  # Higher score indicates better alignment with Golden Ratio

# Function to calculate facial features based on image
def calculate_facial_features(image_path, resize_factor=1.0):
    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return None

    # Resize the image based on the resize factor
    original_height, original_width = image.shape[:2]
    new_width = int(original_width * resize_factor)
    new_height = int(original_height * resize_factor)
    resized_image = cv2.resize(image, (new_width, new_height))

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe Face Mesh
    result = face_mesh.process(rgb_image)

    if not result.multi_face_landmarks:
        print("No face detected!")
        return None

    landmarks = result.multi_face_landmarks[0].landmark

    # Extract image dimensions for pixel calculations
    image_width, image_height = resized_image.shape[1], resized_image.shape[0]

    # Calculate facial symmetry score
    symmetry_score = calculate_symmetry_score(landmarks, image_width, image_height)

    # Calculate Golden Ratio score
    golden_ratio_score = calculate_golden_ratio_score(landmarks, image_width, image_height)

    # Visualize key facial landmarks
    for landmark in landmarks:
        x, y = int(landmark.x * new_width), int(landmark.y * new_height)
        cv2.circle(resized_image, (x, y), 2, (0, 255, 0), -1)

    # Display the image with annotations
    cv2.imshow("Facial Symmetry and Golden Ratio Analysis", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return symmetry_score, golden_ratio_score

# Run the calculation on the uploaded image
image_path = "C:/Users/anshi/Desktop/face-topsis/faces/angelina.jpg"  # Path to the uploaded image
symmetry_score, golden_ratio_score = calculate_facial_features(image_path, resize_factor=0.6)

if symmetry_score is not None:
    print(f"Facial Symmetry Score: {symmetry_score:.2f}")
    print(f"Golden Ratio Score: {golden_ratio_score:.2f}")
