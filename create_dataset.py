from cv2 import data
import mediapipe as mp
import cv2
import os

from face_utils import cal, create_directory, save_image


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

vid = cv2.VideoCapture(0)

while(True):

    rect, frame = vid.read()

    with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    min_detection_confidence=0.2, 
    min_tracking_confidence=0.2) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated_image = frame.copy()
        if not results.multi_face_landmarks:
            continue

        idx_to_coordinates = []
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                l = [landmark.x, landmark.y,landmark.z]
                idx_to_coordinates.append(l)

        ptsold = [idx_to_coordinates[10], idx_to_coordinates[33], idx_to_coordinates[152], idx_to_coordinates[263]]
        angles = cal(ptsold)
        yaw = angles["yaw"]
        print(yaw)
        yaw = int(yaw)
        data_dir = "P{}".format(yaw)
        if yaw < 0:
            data_dir = "N{}".format(yaw)
        save_image(frame, data_dir)
        cv2.putText(frame, str(yaw),(105, 105),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(225,0,0))
        cv2.imshow("frame", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv2.destroyAllWindows()


