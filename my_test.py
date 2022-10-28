import cv2
import mediapipe as mp
# img = cv2.imread('jaundice_eye_color_ranks.jpeg')
#
# print('0-4mg/dL No treatment needed')
# # 557, 88
# print(img[150,70])
# print('5-14mg/dL Phototherapy')
# print(img[290,70])
# print('15-19mg/dL Phototherapy')
# print(img[430,70])
# print('>20mg/dL Phototherapy/Exchange transfusion')
# print(img[570,70])
# [255 255 255] - [217 255 255] 0-4mg/dl no treatment
# [217 255 255] - [184 254 253] 5-14mg/dl phototreatment
# [184 254 253] - [138 254 255] 15-19mg/dl phototreatment
# [138 254 255] - [ 40 254 255] >20mg/dl phototreatment
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
image = cv2.imread('rock.jpg')
height, width, _=image.shape
print('Height, Width',height,width)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#facial landmark form rgb image
result = face_mesh.process(image)

for facial_landmarks in result.multi_face_landmarks:
    pt1 = facial_landmarks.landmark[0]

    x = int((pt1.x)*width)
    y = int((pt1.y)*height)
    cv2.circle(image,(,),15,(0,0,255))


cv2.imshow('image',image)
cv2.waitKey(0)