import cv2
import mediapipe as mp
import os
# from sklearn.preprocessing import normalize
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

file_dir = "data"
file_list = os.listdir(file_dir)
file_list = ['test.jpg']


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

# For static images:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(file_list):
    fPath = os.path.join(file_dir, file)
    image = cv2.imread(fPath)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # print("resss ", results.__dict__)
    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    image_rows, image_cols, _ = annotated_image.shape #hwc
    i = 0
    box = []
    for detection in results.face_detections:
      mp_drawing.draw_detection(annotated_image, detection)
      boxxx = detection.location_data.relative_bounding_box
      box = [boxxx.xmin, boxxx.ymin, boxxx.width, boxxx.height]
    idx_to_coordinates = []
    for face_landmarks in results.multi_face_landmarks:
      for idx, landmark in enumerate(face_landmarks.landmark):
        l = [landmark.x, landmark.y,landmark.z]
        idx_to_coordinates.append(l)
      # print('face_landmarks:', face_landmarks)
      i = i+1
      # print("aaaaaaaaaaaaa", idx_to_coordinates)
      calculateAngle(box, idx_to_coordinates, [image_cols, image_rows])
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)
    cv2.imwrite('datamesh/' + str(file), annotated_image)