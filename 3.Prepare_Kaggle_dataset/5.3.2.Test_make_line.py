import cv2
import numpy as np

img = cv2.imread('D:/0307.jpg') 
b=[734, 743, 745, 746, 745, 743, 734, 734, 725, 723, 722, 723, 725, 734, 383, 387, 391, 395, 399, 403, 407, 407, 403, 399, 395, 391, 387, 383]
print (len(b))

for a in range(0,12,1): 
 cv2.line(img, (b[a], b[a+15]), (b[a+1], b[a+16]), (49,90,180))
cv2.imshow("foo",img)
cv2.waitKey()

