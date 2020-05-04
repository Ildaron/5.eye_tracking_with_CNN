import cv2 
import numpy as np
image_locate="D:/worktable/science/13.Eyes_tracking/3.My_neural_network_for_Eye/9.next_try/1.test/1.2_all_image/new_image/"               
big_radius=[]
#Step_1 
def iris_size(frame):
        height, width = frame.shape
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

def find_best_threshold(eye_frame):
 average_iris_size = 0.14
 trials = {}
 for threshold in range(0, 100, 5):
  iris_frame=cv2.threshold(eye_frame, threshold, 255, cv2.THRESH_BINARY)[1]
  trials[threshold] = iris_size(iris_frame)
 # print (trials.items())
  best_threshold = min(trials.items(), key=(lambda ron: abs(ron[1] - average_iris_size)))
#  print (best_threshold)
 return best_threshold

for count in range (0,9796,1):
 try:
  img=image_locate+str(count)+".jpg"
  img = cv2.imread(img)
  cv2.imshow('ImageWindow1', img)
  first_image=img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  print ("start1")
  threshold = find_best_threshold(img)
  print ("start2")
  img = cv2.threshold(img, int(threshold[0]), 255, cv2.THRESH_BINARY)[1]
  kernel = np.ones((5, 5), np.uint8)
 
  img = cv2.dilate(img,kernel,iterations = 12)
  img = cv2.erode(img,kernel,iterations = 9)
#cv2.imshow('ImageWindow', img) 
  img = img.reshape(416,416, 1)

#Step_2 make cicle
  colorLower = (0)
  colorUpper = (1)
  mask = cv2.inRange(img, colorLower, colorUpper)
  cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
  key = cv2.waitKey(1) & 0xFF
  for c in cnts:   
   ((x, y), radius) = cv2.minEnclosingCircle(c)
   big_radius.append(int(radius))
       
  if radius > 1:
   cv2.circle(first_image, (int(x), int(y)), (max(big_radius)),(0, 255, 255), 2) 
  cv2.imshow("Frame", first_image)
  key = cv2.waitKey(1) & 0xFF
 
  big_radius.clear() 
  print (big_radius)
  print (x,y, radius)
 except ValueError:
  print ("some problem")



