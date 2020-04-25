import cv2
import numpy as np

img = cv2.imread('D:/99.jpg') 
b=[167, 127, 117, 124, 158, 223, 256, 274, 267, 251, 228, 202, 190, 178, 167, 160, 168, 187, 214, 236, 242, 229, 193, 173, 157, 154, 154, 157, 157, 160]
for a in range(0,14,1): 
 cv2.line(img, (b[a], b[a+15]), (b[a+1], b[a+16]), (49,90,180))
cv2.imshow("foo",img)
cv2.waitKey()

