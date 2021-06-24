import cv2
import os
import math
from typing import List, Tuple, Union
import numpy as np

import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

file_dir = "data"
file_list = os.listdir(file_dir)
# file_list = ['R4.jpg', 'R5.jpg', 'Rakesh_1.jpg', 'Rakesh_2.jpg', 'Rakesh_3.jpg']
file_list = ['demo.jpg']

print(file_list)
printAllLandMark = True
outputDir = 'databl/'

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
    max(0, (normalized_x * image_width))
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


# For static images:
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
  for idx, file in enumerate(file_list):
    fPath = os.path.join(file_dir, file)
    image = cv2.imread(fPath)
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print("result ", results.__dict__)

    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      mp_drawing.draw_detection(annotated_image, detection)

      image_rows, image_cols, _ = annotated_image.shape
      location = detection.location_data

      relative_bounding_box = location.relative_bounding_box
      rect_start_point = _normalized_to_pixel_coordinates(
      relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
      image_rows)
      rect_end_point = _normalized_to_pixel_coordinates(
      relative_bounding_box.xmin + relative_bounding_box.width,
      relative_bounding_box.ymin + +relative_bounding_box.height, image_cols,
      image_rows)

      print("rect_start_point ", rect_start_point)
      print("rect_end_point ", rect_end_point)

      # i = 0
      # location = detection.location_data
      # for keypoint in location.relative_keypoints:
      #   if(not (printAllLandMark) and i > 3):
      #     continue
      #   keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
      #                                              image_cols, image_rows)
      #   cv2.circle(annotated_image, keypoint_px,2,
      #          (0, 255, 0), 2)
      #   i = i + 1

    cv2.imwrite(outputDir + str(file), annotated_image)


