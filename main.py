import cv2
import mediapipe as mp
import numpy as np
from visualizer import Visualizer
from mesh import Mesh
import open3d as o3d
# Initialize Mediapipe face detection and mesh models

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

vis = Visualizer()
mesh = Mesh('mask.obj')

# Function to draw the 3D mesh on the face
def draw_3d_mesh(image, landmarks):
    # Your code to draw the 3D mesh on the face goes here
    #mp_drawing.draw_landmarks(image, landmarks)
    pass

# Main function for face detection and 3D mesh projection
def main():
    cap = cv2.VideoCapture(1)  # Change to the desired video source if not using a webcam

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        # BACKGROUND

        # Set up Open3D camera intrinsic parameters
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # width, height, _ = image.shape
        
        intrinsic.set_intrinsics(width, height, 35, 30, width // 2, height // 2)
        frame = cv2.flip(image, 0)
        frame = cv2.flip(frame, 1)
        # Convert the frame to RGB (Open3D expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create an Open3D image from the frame
        img_o3d = o3d.geometry.Image(frame_rgb)

        # Create a synthetic depth image (set all depth values to 1.0)
        depth_data = np.ones((frame.shape[0], frame.shape[1]), dtype=np.float32)
        depth_image = o3d.geometry.Image(depth_data)

        # Create an Open3D RGBD image from the color and depth images
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            img_o3d, depth_image, depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )

        # Create a point cloud from the RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        vis.add_object(pcd)
        # vis._vis.update_geometry(pcd)
        # Convert the image from BGR to RGB (Mediapipe expects RGB input)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        results = face_mesh.process(image_rgb)
        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                cam_matrix = np.array([[img_w, 0, img_h / 2],
                                       [0, img_w, img_w / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                if not success:
                    print("Failed to estimation pose!")
                    continue
                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                Rx = angles[0] * 360
                Ry = angles[1] * 360
                Rz = angles[2] * 360

                Tx = trans_vec[0][0]
                Ty = trans_vec[1][0]
                Tz = trans_vec[2][0]

                print("Rotation: ", [Rx, Ry, Rz])
                print("Translation: ", [Tx, Ty, Tz])

                mesh.transform([Tx, Ty, Tz], [Rx, Ry, Rz])
                vis.add_object(mesh._mesh)


        vis.show()

    vis.clear()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
