import math
import os
from datetime import datetime

import cv2

data_dir = os.path.join(os.getcwd(), "face_dataset")

def radians(a1, a2, b1, b2):
  return math.atan2(b2 - a2, b1 - a1)

def cal(mesh): #10, 33, 152, 263
  roll = math.degrees(radians(mesh[1][0], mesh[1][1], mesh[3][0], mesh[3][1]))
  yaw =math.degrees( radians(mesh[1][0], mesh[1][2], mesh[3][0], mesh[3][2]))
  pitch= math.degrees(radians(mesh[0][1], mesh[0][2], mesh[2][1], mesh[2][2]))
  # print(f"yaw {yaw}, roll {roll}, pitch {pitch}")
  # print(f"yaw {(yaw)}, roll {(roll)}, pitch {(pitch)}")
  return {"yaw": yaw, "roll":roll, "pitch":pitch}

def create_directory(dir_path: str):
    current_dir = os.path.join(data_dir, dir_path)

    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    
    return current_dir

def save_image(image, data_dir):
    current_dir = create_directory(data_dir)    
    dt = datetime.now()
    micros = dt.microsecond
    image_name = "{0}.{1}".format(micros, "jpeg")
    image_path = os.path.join(current_dir, image_name)
    cv2.imwrite(image_path, image)