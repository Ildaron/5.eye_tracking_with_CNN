from keras.models import load_model
import glob
import numpy as np
import PIL
from PIL import Image
import cv2

dataset_amount=255
size_image=416

#1_step1 - load data X
imageFolderPath = 'D:/train/'
imagePath = glob.glob(imageFolderPath + '/*.JPG') 
im_array = np.array( [np.array(Image.open(img).convert('L'), 'f') for img in imagePath] )
im_array=im_array/255
im_array = im_array.reshape(dataset_amount,size_image,size_image, 1)

i = int(0.8 * dataset_amount)
test_X = im_array[i:]

#2_test2-load model
model = load_model('ildar_eye.h5')
model.summary() 

#3_stet3-predict_Y
pred_y = model.predict(test_X)   #test_X
#print (test_X.shape)
#print (pred_y.shape)

b=pred_y[27]
#print(type(b))
#img=test_X[27]
#4- step4 predict video

cap = cv2.VideoCapture("D:/project/video.avi") 
while 1:
 
 ret, frame = cap.read() 
 #cv2.imshow('frame',frame) 
#if cv2.waitKey(1) & 0xFF == ord('q'): 
# break  
 img=frame

 name="D:/image_test/1.jpg"
 #img = cv2.cvtColor(name, cv2.COLOR_BGR2GRAY)
 img=cv2.imwrite(name,img)

 imageFolderPath = 'D:/worktable/science/13.Eyes_tracking/3.My_neural_network_for_Eye/7._START_project/image_test/'
 imagePath_gray=glob.glob(imageFolderPath + '/*.JPG')
 print(imagePath_gray)
 #im_array = np.array( [np.array(Image.open(img).convert('L'), 'f') for img in imagePath_gray] ) # works zdest
 img_for_video = cv2.imread('D:/worktable/science/13.Eyes_tracking/3.My_neural_network_for_Eye/7._START_project/image_test/1.jpg')
 img_for_video = cv2.cvtColor(img_for_video, cv2.COLOR_BGR2GRAY)
 im_array=img_for_video
 print(im_array.shape)
 img= im_array.reshape(1,416,416,1)
# print(img.shape) 
 pred_y = model.predict(img) 
 b=pred_y[0]
 #print("ild2")

# print (img.shape)
 img_video=img.reshape(416,416)
# print (img_video.shape)
 for a in range(0,14,1): 
  cv2.line(img_video, (b[a], b[a+15]), (b[a+1], b[a+16]), (49,90,180))
 cv2.imshow("image",img_video)
 cv2.waitKey(50)

# test
#cap = cv2.VideoCapture('video.avi')
