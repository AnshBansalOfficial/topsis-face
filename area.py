import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# MESH_ANNOTATIONS with proper indices for facial landmarks
MESH_ANNOTATIONS = {
    "leftnoseupper": [8, 1, 49, 8],
    "rightnoseupper": [8, 1, 279, 8],
    "noselower": [49, 1, 279, 2, 49],
}

# Function to calculate the area of a polygon using the Shoelace formula
def calculate_polygon_area(points):
    n = len(points)
    area = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - y1 * x2
    return abs(area) / 2

# Function to process an image and generate face landmarks and a face mask
def process_face_image(image_path, scale_factor=0.5):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image. Check the file path.")
        return

    # Convert the image to RGB (MediaPipe works with RGB images)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect face landmarks
    results = face_mesh.process(rgb_image)

    # Check if any faces were detected
    if not results.multi_face_landmarks:
        print("No face detected in the image.")
        return

    # Get the first detected face's landmarks
    face_landmarks = results.multi_face_landmarks[0]

    # Draw contours and calculate areas
    points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
              for landmark in face_landmarks.landmark]

    for region, indices in MESH_ANNOTATIONS.items():
        polygon_points = [points[idx] for idx in indices]
        # Draw the polygon on the image
        for i in range(len(indices) - 1):
            pt1 = points[indices[i]]
            pt2 = points[indices[i + 1]]
            cv2.line(image, pt1, pt2, (0, 255, 255), 2)  # Yellow line for contours
        
        # Calculate the area of the polygon
        area = calculate_polygon_area(polygon_points)
        
        # Annotate the area on the image
        centroid_x = int(np.mean([p[0] for p in polygon_points]))
        centroid_y = int(np.mean([p[1] for p in polygon_points]))
        cv2.putText(image, f"Area: {area:.2f}", (centroid_x, centroid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        print(f"Region: {region}, Area: {area:.2f}")

    # Resize the image for display
    resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

    # Display the window
    cv2.imshow(f"Face Contours with Areas (Scale: {scale_factor}x)", resized_image)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the path to the image
image_path = "C:/Users/anshi/Desktop/face-topsis/faces/angelina.jpg"  # Replace with your image path
process_face_image(image_path, 1.5)
