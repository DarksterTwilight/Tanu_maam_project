import cv2

# img  = cv2.imread('shutterstock.jpg')
#
# scr_img = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
#
# print('Scr img shape')
# print(scr_img.shape)
# print('Lightest color value')
# print(scr_img[384, 233])
#
# print('darkest color value')
#print(scr_img[267, 654])

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
  max_num_faces= 1,
  refine_landmarks= True,
  min_detection_confidence= 0.5,
  min_tracking_confidence=0.5) as face_mesh:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("No camera found")

      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        my_facemesh_countours = mp_drawing.draw_landmarks(
          image=image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())

        my_irises = mp_drawing.draw_landmarks(
          image = image,
          landmark_list = face_landmarks,
          connections = mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec =None,
          connection_drawing_spec = mp_drawing_styles
          .get_default_face_mesh_iris_connections_style()
        )




    cv2.imshow('Image', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
print('Results')
#print(results.shape)
print(results)

cap.release()



