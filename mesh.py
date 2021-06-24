import cv2
import mediapipe as mp
import os
from typing import List, Tuple, Union

# from sklearn.preprocessing import normalize
import math
import numpy as np
from datetime import datetime
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

file_dir = "data"
file_list = os.listdir(file_dir)
file_list = ['demo.jpg']#1616835294314.JPEG']
# file_list.extend(file_list)
# file_list.extend(file_list)
# for i in range(100):
#   file_list.append("nn.JPEG")

RIGHT_EAR = 234
NOSE = 5
CHIN_END = 152
LEFT_EAR = 454


def rotationMatrixToEulerAngle(r):
  [r00, r01, r02, r10, r11, r12, r20, r21, r22] = r
  if (r10 < 1):
    if (r10 > -1):
      thetaZ = math.asin(r10)
      thetaY = math.atan2(-r20, r00)
      thetaX = math.atan2(-r12, r11)
    else:
      thetaZ = -math.PI / 2
      thetaY = -math.atan2(r21, r22)
      thetaX = 0
  else:
    thetaZ = math.PI / 2
    thetaY = math.atan2(r21, r22)
    thetaX = 0
  return { "pitch": 2 * -thetaX, "yaw": 2 * -thetaY, "roll": 2 * -thetaZ }

def normalize(v):
  length = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
  vq = [v[0]/length, v[1]/length, v[2]/length]
  # v[0] = v[0]/length
  # v[1] = v[1]/length
  # v[2] = v[2]/length
  return vq

def calculateAngle(boxRaw, mesh, image_size):
  print("image_sizeimage_sizeimage_sizeimage_size ", image_size)

  size = max(boxRaw[2] * image_size[0], boxRaw[3] * image_size[1])/ 1.5
  print("sizeeeeeeeeeeeeeeee ", size)

  ptsold = [mesh[10], mesh[152], mesh[234], mesh[454]]
  pts = []
  print("ptss ",ptsold)
  for pt in ptsold:
    item = [pt[0]*image_size[0]/size, pt[1]*image_size[1]/size, pt[2]]
    pts.append(item)
  print(pts)

  y_axis = normalize(np.subtract(pts[1], pts[0]))
  x_axis = normalize(np.subtract(pts[3], pts[2]))
  z_axis = normalize(np.cross(x_axis, y_axis))
  x_axis = np.cross(y_axis, z_axis)

  matrix = [
    x_axis[0], x_axis[1], x_axis[2],
    y_axis[0], y_axis[1], y_axis[2],
    z_axis[0], z_axis[1], z_axis[2],
  ]
  angle = rotationMatrixToEulerAngle(matrix)
  print("angggggggggggggg ", angle)


def radians(a1, a2, b1, b2):
  return math.atan2(b2 - a2, b1 - a1)

def cal(mesh): #10, 33, 152, 263
  roll = math.degrees(radians(mesh[1][0], mesh[1][1], mesh[3][0], mesh[3][1]))
  yaw =math.degrees( radians(mesh[1][0], mesh[1][2], mesh[3][0], mesh[3][2]))
  pitch= math.degrees(radians(mesh[0][1], mesh[0][2], mesh[2][1], mesh[2][2]))
  # print(f"yaw {yaw}, roll {roll}, pitch {pitch}")
  # print(f"yaw {(yaw)}, roll {(roll)}, pitch {(pitch)}")
  return {"yaw": yaw, "roll":roll, "pitch":pitch}


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

# For static images:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,
    min_detection_confidence=0.2, 
    min_tracking_confidence=0.2) as face_mesh:
  dd = {}
  for idx, file in enumerate(file_list):
    fPath = os.path.join(file_dir, file)
    image = cv2.imread(fPath)
    a = datetime.now()
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print(f"{file} ntimeeeeeeee ", datetime.now()-a)
    # print("resss ", results.__dict__)
    # Print and draw face mesh landmarks on the image.
    print("facesssssssss ", len(results.multi_face_landmarks))

    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    image_rows, image_cols, _ = annotated_image.shape #hwc
    i = 0
    box = []
    # for detection in results.multi_face_landmarks:
    #   # mp_drawing.draw_detection(annotated_image, detection)
    #   boxxx = detection.location_data.relative_bounding_box
    #   box = [boxxx.xmin, boxxx.ymin, boxxx.width, boxxx.height]
    idx_to_coordinates = []
    for face_landmarks in results.multi_face_landmarks:
      for idx, landmark in enumerate(face_landmarks.landmark):
        l = [landmark.x, landmark.y,landmark.z]
        idx_to_coordinates.append(l)
      # print('face_landmarks:', face_landmarks)
      i = i+1
      # print("aaaaaaaaaaaaa", idx_to_coordinates)
      # calculateAngle(box, idx_to_coordinates, [image_cols, image_rows])

      # ptsold = [idx_to_coordinates[10], idx_to_coordinates[33], idx_to_coordinates[152], idx_to_coordinates[263]]
      ptsold = [idx_to_coordinates[LEFT_EAR], idx_to_coordinates[NOSE], idx_to_coordinates[RIGHT_EAR], idx_to_coordinates[CHIN_END]]
      dd[file] = cal(ptsold)
      aaa = []
      for pt in ptsold:
        rr = _normalized_to_pixel_coordinates(abs(pt[0]), abs(pt[1]),image_cols, image_rows )
        if(file == "R4.jpg"):
          print(f"aaaaaaaaaaaa {pt}")
          print("rr ", rr)
        keypoint_px, keypoint_py = rr
        cord = (keypoint_px, keypoint_py, pt[2])
        aaa.append(cord)
        cv2.circle(annotated_image, (keypoint_px, keypoint_py),2,(255, 255, 255), 2)
      # cal(aaa)
      # roll = math.degrees(radians(aaa[1][0], aaa[1][1], aaa[3][0], aaa[3][1]))
      # print("dddddddd ", roll)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)
    cv2.imwrite('datamesh/' + str(file), annotated_image)
  print("dddddddddd ", dd)