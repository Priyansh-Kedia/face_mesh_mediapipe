import mediapipe as mp
import cv2
import os

from face_utils import cal

os.environ["tflite"] ="2"

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

image = cv2.imread("test.jpg")
IMAGE_FILES = ["test.jpg"]
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.8) as face_detection:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      print('Nose tip:')
      print(mp_face_detection.get_key_point(
          detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)
    cv2.imwrite('annotated_image' + str(idx) + '.png', annotated_image)

with mp_face_mesh.FaceMesh(
    static_image_mode = True,
    max_num_faces = 10,
    min_detection_confidence = 0.1,
    min_tracking_confidence = 0.1) as face_mesh:

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()
    # cv2.imshow("image",image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    if results.multi_face_landmarks:
        print(len(results.multi_face_landmarks))
        idx_to_coordinates = []
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                l = [landmark.x, landmark.y,landmark.z]
                idx_to_coordinates.append(l)
            ptsold = [idx_to_coordinates[10], idx_to_coordinates[33], idx_to_coordinates[152], idx_to_coordinates[263]]
            angles = cal(ptsold)
            print(angles)
            idx_to_coordinates = []

            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            

        

    cv2.imwrite("test1.png", annotated_image)
