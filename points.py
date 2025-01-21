import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Load an image
image = cv2.imread("C:/Users/anshi/Desktop/face-topsis/faces/angelina.jpg")  # Replace with your image path
resize_factor = 1.5  # Set the resize factor (e.g., 0.5 for 50% smaller)
image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
image_height, image_width, _ = image.shape

# Convert image to RGB for MediaPipe
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to get landmarks
results = face_mesh.process(rgb_image)

# Check if landmarks are detected
if results.multi_face_landmarks:
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
        # Draw landmarks
        for idx, landmark in enumerate(face_landmarks.landmark):
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            if idx in (146, 91, 405, 321, 375, 291,61, 185, 40, 39, 409, 291):  # Highlight specific points
                cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)  # Red circle
                cv2.putText(annotated_image, str(idx), (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.circle(annotated_image, (x, y), 2, (255, 255, 255), -1)  # White dot

    # Display the image
    cv2.imshow("Landmarks with Indices", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face landmarks detected.")
