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

            # Define regions for the facial mask
            points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                      for landmark in face_landmarks.landmark]

            # Define regions for the facial mask
            jaw = points[0:17]  # Jawline
            left_eyebrow = points[17:22]  # Left eyebrow
            right_eyebrow = points[22:27]  # Right eyebrow
            nose_bridge = points[27:31]  # Nose bridge
            lower_nose = points[31:36]  # Lower part of the nose
            left_eye = points[36:42]  # Left eye
            right_eye = points[42:48]  # Right eye
            outer_lip = points[48:60]  # Outer lip
            inner_lip = points[60:68]  # Inner lip
            forehead = points[8:9]  # Single point for the forehead (can be expanded)
            cheeks = points[1:5] + points[13:17]  # Cheeks (left and right side)
            nose_tip = points[33:34]  # Nose tip
            chin = points[5:11]  # Chin area

            # Define expanded regions for the face mask
            mask_regions = [
                jaw, left_eyebrow, right_eyebrow, nose_bridge, lower_nose,
                left_eye, right_eye, outer_lip, inner_lip,
                forehead, cheeks, nose_tip, chin
            ]

            # Create a separate mask frame (black background)
            mask_frame = np.zeros_like(frame)

            # Draw lines to connect points and create the face mask
            for region in mask_regions:
                for i in range(len(region) - 1):
                    cv2.line(mask_frame, region[i], region[i + 1], (0, 255, 255), 2)  # Draw lines in yellow
                # Connect the last point to the first to close the loop for circular regions
                if region in [left_eye, right_eye, outer_lip, inner_lip]:
                    cv2.line(mask_frame, region[-1], region[0], (0, 255, 0), 2)  # Green for closing loops

            # Display the face mask in a new window
            cv2.imshow("Face Mask", mask_frame)

    # Display the original frame (with landmarks)
    cv2.imshow("Original Frame with Landmarks", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
