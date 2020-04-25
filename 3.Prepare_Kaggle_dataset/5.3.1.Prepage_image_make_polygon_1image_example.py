import cv2
import numpy as np
import math

image= cv2.imread('D:/worktable/science/13.Eyes_tracking/3.My_neural_network_for_Eye/0.1.Datasets/datasets/3/eye-gaze/MPIIGaze/Data/Original/p00/day14/0307.jpg')
#input data
x =734
y = 395
w = 10
h =50
x_0=734
y_0=395
radius=12
polygon=[]
polygon_negative=[]
polygon_for_model_X=[]
polygon_for_model_Y=[]
polygon_for_model_X_Y=[]
# start calculation

for y in range (-radius,radius+1,4): # (-radius,radius+1,2):
 x=math.sqrt((radius*(radius)-y*y))
 negative_x=x
 negative_x*= -1 

 polygon.append(round(x+734))
 polygon.append(y+395)
 
 polygon_negative.append(y+395)
 polygon_negative.append(round(negative_x+734))
 
polygon_negative.reverse()

for a in polygon_negative:
 polygon.append(a)
 
z=0
#image = cv2.circle(image, (x_0,y_0), radius, (255, 0, 0) , 1)
for a in range(0,25,2):  # 34
 cv2.line(image, (polygon[a], polygon[a+1]), (polygon[a+2], polygon[a+3]), (49,90,180))
 z=z+1
# print (z)
image=image[y_0-30:y_0+30, x_0-40:x_0+40]
newsize = (416, 416)
image=cv2.resize(image,newsize)
#cv2.imshow("foo",image)
#cv2.waitKey()

for a in range (0, len(polygon),2):
# print (a)
 polygon_for_model_X.append(polygon[a])
 polygon_for_model_Y.append(polygon[a+1])


print ("polygon", polygon)
print ("polygon_for_model_X", polygon_for_model_X)
print ("polygon_for_model_Y", polygon_for_model_Y)

polygon_for_model_X_Y=polygon_for_model_X
for a in polygon_for_model_Y:
 polygon_for_model_X_Y.append(a)

print ("polygon_for_model_X_Y", polygon_for_model_X_Y)
