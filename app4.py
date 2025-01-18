import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to process an image and generate face landmarks and a face mask
def process_face_image(image_path):
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

    # Draw landmarks on the original image
    image_with_landmarks = image.copy()
    mp_drawing.draw_landmarks(
        image_with_landmarks, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=10, circle_radius=0),  # Solid dots
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=10, circle_radius=0)   # Solid dots
    )

    # Create a blank mask frame (black background)
    mask_frame = np.zeros_like(image)

    # Extract the points from the landmarks
    points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
              for landmark in face_landmarks.landmark]

    # Debug: Print the computed points
    print("Points:", points)

    # Define the exact facial landmarks for the face contours
    face_contours = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # Jawline
        [17, 18, 19, 20, 21],  # Left eyebrow
        [22, 23, 24, 25, 26],  # Right eyebrow
        [27, 28, 29, 30],  # Nose bridge
        [31, 32, 33, 34, 35],  # Lower nose
        [36, 37, 38, 39, 40, 41],  # Left eye
        [42, 43, 44, 45, 46, 47],  # Right eye
        [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],  # Outer lip
        [60, 61, 62, 63, 64, 65, 66, 67],  # Inner lip
    ]

    # Debug: Print the face contours
    print("Face Contours:", face_contours)

    # Draw the face mask using yellow lines
    for contour in face_contours:
        for i in range(len(contour) - 1):
            pt1 = points[contour[i]]
            pt2 = points[contour[i + 1]]
            # Ensure points are within valid range
            if 0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and \
               0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]:
                cv2.line(mask_frame, pt1, pt2, (0, 255, 255), 2)

    # Optionally, use predefined MediaPipe connections for contours
    mp_drawing.draw_landmarks(
        mask_frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=0)  # Solid lines
    )

    # Resize the windows for display
    scale_factor = 0.25  # Adjust the scale factor as needed
    resized_landmarks = cv2.resize(image_with_landmarks, (0, 0), fx=scale_factor, fy=scale_factor)
    resized_mask = cv2.resize(mask_frame, (0, 0), fx=scale_factor, fy=scale_factor)

    # Display the windows
    cv2.imshow("468 Landmarks on Face", resized_landmarks)
    cv2.imshow("Face Mask", resized_mask)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the path to the image
image_path = "C:/Users/anshi/Desktop/face-topsis/face.jpg"  # Replace with your image path
process_face_image(image_path)
