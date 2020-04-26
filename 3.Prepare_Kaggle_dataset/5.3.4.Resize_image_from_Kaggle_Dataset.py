import numpy as np
import cv2
import os

label = open('D:/all_annotation.txt', 'r')
label=label.read()
label=label.split()
dataset=np.array(label)
dataset=dataset.reshape(14763,41,1)
dataset = np.delete(dataset, [0,1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41],1)  # delete second row of A
#print (dataset.shape)
#print (dataset[0,0])
#print (dataset[0,1])
newsize = (416, 416)

for a in range (1,11221,1):
 image_place="D:/rename_folder_dataset/"+str(a)+".jpg"
 image= cv2.imread(image_place)
 x=int(dataset[a-1,0])
 y=int(dataset[a-1,1])
# print (x)
 print (a)
 #image = cv2.circle(image, (dataset[0,0],dataset[0,1]), 10, (255, 0, 0) , 1) 
 image=image[y-30:y+30, x-40:x+40]
 image=cv2.resize(image,newsize)
 new_image_place="D:/resize_image/new_image/"
#cv2.imwrite(os.path.join(new_image_place, 'waka.jpg'), new_image_place)
 cv2.imwrite(os.path.join("D:/new_image/" ,  str(a)+".jpg"), image)
 #cv2.imshow("foo",image)
 cv2.waitKey()



