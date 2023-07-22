import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt  # Optional, for visualization

# Initialize Mediapipe face detection and mesh models
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Function to draw the 3D mesh on the face
def draw_3d_mesh(image, landmarks):
    # Your code to draw the 3D mesh on the face goes here
    mp_drawing.draw_landmarks(image, landmarks)
    #pass

# Main function for face detection and 3D mesh projection
def main():
    cap = cv2.VideoCapture(0)  # Change to the desired video source if not using a webcam

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
         mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            # Convert the image from BGR to RGB (Mediapipe expects RGB input)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces in the image
            results_detection = face_detection.process(image_rgb)
            if results_detection.detections:
                for detection in results_detection.detections:
                    # Extract face landmarks
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    face_region = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                    results_mesh = face_mesh.process(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
                    if results_mesh.multi_face_landmarks:
                        for face_landmarks in results_mesh.multi_face_landmarks:
                            # Draw 3D mesh on the face
                            draw_3d_mesh(face_region, face_landmarks)

            # Show the image with the 3D mesh (optional, for visualization)
            cv2.imshow('Face Mesh', image)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the loop
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
