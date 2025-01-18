import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start video capture from the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Unable to access the camera")
    exit()

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB (MediaPipe works with RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect faces
    results = face_mesh.process(rgb_frame)

    # Check if any faces were detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face landmarks in the original frame (for reference)
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # Create a frame for drawing the mask (black background)
            mask_frame = np.zeros_like(frame)

            # Define the points for the face structure (landmarks)
            points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                      for landmark in face_landmarks.landmark]

            # Define the exact facial landmarks for the face contours (as used in MediaPipe)
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

            # Draw the connections (lines) for the face mask using yellow color
            for contour in face_contours:
                for i in range(len(contour) - 1):
                    # Connect each point to the next one
                    cv2.line(mask_frame, points[contour[i]], points[contour[i + 1]], (0, 255, 255), 2)

            # Display the mask frame (highlighting the face structure with yellow lines)
            cv2.imshow("Face Mask", mask_frame)

    # Display the original frame with landmarks
    cv2.imshow("Original Frame with Landmarks", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
