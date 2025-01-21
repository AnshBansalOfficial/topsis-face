import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# MESH_ANNOTATIONS with proper indices for facial landmarks
MESH_ANNOTATIONS = {
    "silhouette": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,10],
    "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,61],
    "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291,61,185],
    "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,78],
    "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,78],
    "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
    "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
    "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
    "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
    "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
    "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
    "rightEyebrowLower": [35, 124, 46, 53, 52, 65],
    "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
    "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
    "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
    "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
    "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
    "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
    "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
    "leftEyebrowLower": [265, 353, 276, 283, 282, 295],
    "midwayBetweenEyes": [168],
    "noseTip": [1],
    "noseBottom": [2],
    "noseRightCorner": [98],
    "noseLeftCorner": [327],
    "rightCheek": [205],
    "leftCheek": [425],

    "leftnoseupper":[8,1,49,8],
    "rightnoseupper":[8,1,279,8],
    "noselower":[49,1,279,2,49],

    "leftCheekbone": [227, 147, 214, 210,170,149],
    "rightCheekbone": [447, 376, 434, 430, 395, 378],

    "lefteyetolip":[226,61],
    "righteyetolip":[291,446],

    "foreheadmidmid":[9,107,55,8,285,336,9],
    "foreheadmidmidleft":[151,9,107,69,67,109,10,151],
    "foreheadmidmidright":[151,9,336,299,297,338,10,151],
    "foreheadleft":[54,70,63,105,66,107,69,67,103,54],
    "foreheadright":[284,300,293,334,296,336,299,297,332,284]

}

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

    # Draw contours on the original image
    points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
              for landmark in face_landmarks.landmark]

    for region, indices in MESH_ANNOTATIONS.items():
        for i in range(len(indices) - 1):
            pt1 = points[indices[i]]
            pt2 = points[indices[i + 1]]
            # Ensure points are within valid range
            if 0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and \
               0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]:
                cv2.line(image, pt1, pt2, (0, 255, 255), 2)  # Yellow line for contours

    # Resize the image for display
    resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

    # Display the window
    cv2.imshow(f"Face Contours (Scale: {scale_factor}x)", resized_image)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the path to the image
image_path = "C:/Users/anshi/Desktop/face-topsis/faces/angelina.jpg"  # Replace with your image path
process_face_image(image_path, 1.5)
