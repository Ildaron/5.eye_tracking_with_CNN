import cv2
import numpy as np
import math

image= cv2.imread('D:/0307.jpg')
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
 
#print (len(polygon))
z=0
#image = cv2.circle(image, (x_0,y_0), radius, (255, 0, 0) , 1)
for a in range(0,25,1):  # 34
 cv2.line(image, (polygon[a], polygon[a+1]), (polygon[a+2], polygon[a+3]), (49,90,180))
 z=z+1
# print (z)
image=image[y_0-30:y_0+30, x_0-40:x_0+40]
newsize = (416, 416)
image=cv2.resize(image,newsize)
cv2.imshow("foo",image)
cv2.waitKey()

print ((polygon))
