#%% Import section
import numpy as np
import cv2
#%% Pretrained dataset
face_cascade_name = 'data/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
#%%
tests_path = 'TestImages/rear/'

#%% HarrCascade init
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

#%% Loading pretrained
#-- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
#%%
test_image = cv2.imread(tests_path + 't-pantaloni-rearCIC.jpg', cv2.IMREAD_GRAYSCALE)
#%% histogram equalization
frame_gray = cv2.equalizeHist(test_image)
#-- Detect faces
faces = face_cascade.detectMultiScale(frame_gray)
for (x,y,w,h) in faces:
    center = (x + w//2, y + h//2)
    frame = cv2.ellipse(test_image, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    faceROI = frame_gray[y:y+h,x:x+w]
    #-- In each face, detect eyes

cv2.imshow('Capture - Face detection', test_image)