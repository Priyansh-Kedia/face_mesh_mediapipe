import os
import json 
from json import JSONEncoder

import mediapipe as mp
import cv2

from face_utils import cal

# Gives a std dev of 3.683001728129547

class Data():
    def __init__(self, value, diff):
        self.value = value
        self.diff = diff

class DataEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__

data_dir = os.path.join(os.getcwd(), "face_dataset")
# current_dir = os.path.join(data_dir)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

mean_list = {}
for directory in os.listdir(data_dir):
     angles_list = []
     for file in os.listdir(os.path.join(data_dir, directory)):
        with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        min_detection_confidence=0.2, 
        min_tracking_confidence=0.2) as face_mesh:
            file_path = os.path.join(data_dir, directory, file)
            frame = cv2.imread(file_path)
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
            angles_list.append(int(angles["yaw"]))

     mean_list[directory] = Data(sum(angles_list)/len(angles_list), float(directory[1:]) - sum(angles_list)/len(angles_list))
     angles_list = []

     print(directory, file)
    
# print(json.dumps(mean_list, indent=4, cls=DataEncoder))

std_dev = 0
print(type(mean_list))
for value in mean_list.values():
    diff = value.diff
    if diff < 0:
        diff = -diff
    std_dev += diff
    

print(std_dev/len(mean_list))
