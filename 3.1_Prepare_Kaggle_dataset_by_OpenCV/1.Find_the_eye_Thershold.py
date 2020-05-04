import cv2 
import numpy as np
img = cv2.imread('D:/worktable/science/13.Eyes_tracking/3.My_neural_network_for_Eye/8.Train_the_model_by_Kaggle/8.1_prepare_dataset/1.Resize_image/new_image/70.jpg') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('ImageWindow1', img) 

def iris_size(frame):
      #  frame = frame[5:-5, 5:-5]
      #  height, width = frame.shape[:2]
        height, width = frame.shape
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        #print (cv2.countNonZero(frame))
        return nb_blacks / nb_pixels

def find_best_threshold(eye_frame):
 average_iris_size = 0.1
 trials = {}
 for threshold in range(0, 100, 5):
  iris_frame=cv2.threshold(eye_frame, threshold, 255, cv2.THRESH_BINARY)[1]
  trials[threshold] = iris_size(iris_frame)
  print (trials.items())
  best_threshold = min(trials.items(), key=(lambda ron: abs(ron[1] - average_iris_size)))
  print (best_threshold)
 return best_threshold

threshold = find_best_threshold(img)
img = cv2.threshold(img, int(threshold[0]), 255, cv2.THRESH_BINARY)[1]
kernel = np.ones((5, 5), np.uint8)

img = cv2.dilate(img,kernel,iterations = 12)
img = cv2.erode(img,kernel,iterations = 9)
cv2.imshow('ImageWindow', img) 
