from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import numpy as np                                                                                                                             
import glob
import pandas as pd
import PIL
from PIL import Image
import cv2

dataset_amount=255
size_image=416



#step1-make dataset for X-train images
imageFolderPath = 'D:/train/'
imagePath = glob.glob(imageFolderPath + '/*.JPG') # ??
im_array = np.array( [np.array(Image.open(img).convert('L'), 'f') for img in imagePath] )

im_array=im_array/255
im_array = im_array.reshape(dataset_amount,size_image,size_image, 1)
df = pd.read_csv("E:/my_dataset_polygon_all.csv")

df=df["region_shape_attributes"]
num_image=len(df)
dataset_y=np.zeros([])

for a in range(0,num_image,1):
 df_dict=eval(df[a])
 x= df_dict["all_points_x"]
 y = df_dict["all_points_y"]
 dataset_y=np.append(dataset_y,x)
 dataset_y=np.append(dataset_y,y)

dataset_y=np.delete(dataset_y, 0)
dataset_y=dataset_y.reshape(num_image,30)
print (dataset_y.shape)

# step 3 - divide dataset
i = int(0.8 * dataset_amount)
train_X = im_array[:i]
test_X = im_array[i:]
train_y = dataset_y[:i]
test_y = dataset_y[i:]

#step4 - model for neural networks

import matplotlib.pyplot as plt
import numpy as np
import math
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras import layers
from keras import models
import matplotlib.pyplot as plt
import matplotlib
#import pillow
import os
import cv2


model = models.Sequential()
#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)))

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size_image, size_image, 1)))
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)))  #img_size          
model.add(layers.Conv2D(32, (3, 3), activation='relu'))  
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(32, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(600, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(100, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(y.shape[-1], activation='sigmoid'))
#model.add(layers.Dense(y.shape[-1]))

model.add(layers.Dense(30))
#print ("model_finish")
model.summary()
model.compile('adadelta', 'mse', metrics=['acc'])


# step5 - start training 
history = model.fit(train_X, train_y, nb_epoch=800, validation_data=(test_X, test_y), verbose=2)
print ("ok3")

# step6 - to see the result
pred_y = model.predict(test_X)
print (pred_y)

img = cv2.imread('1.jpg') 
b=pred_y[0]
print(type(b))
for a in range(0,14,1): 
 cv2.line(img, (b[a], b[a+15]), (b[a+1], b[a+16]), (49,90,180))
cv2.imshow("foo",img)
cv2.waitKey()


# step7 - to see the graph with accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)                          
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc') 
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss') 
plt.legend()
#plt.show()                        

# step8 - Image Data Generator 

# step-9 - to see between the layers

from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 
activations = activation_model.predict(test_X)
first_layer_activation = activations[0]
#plt.matshow(first_layer_activation[0, :, :, 2], cmap='viridis')
#plt.title('vizualization') 
#plt.show()    

#step-10 to see the all layers
layer_names = [] 
for layer in model.layers[:6]: 
 layer_names.append(layer.name) 
images_per_row = 16
print (layer_names)

for layer_name, layer_activation in zip(layer_names, activations):
 n_features = layer_activation.shape[-1]
 size = layer_activation.shape[1] 
 n_cols = n_features // images_per_row
 display_grid = np.zeros((size * n_cols, images_per_row * size))
 for col in range(n_cols): 
  for row in range(images_per_row):
   channel_image = layer_activation[0, :, :,  col * images_per_row + row]
   channel_image -= channel_image.mean()
   #channel_image /= channel_image.std()
   #channel_image *= 64
   #channel_image += 128
   #channel_image = np.clip(channel_image, 0, 255).astype('uint8')
   #display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image 

# scale = 1. / size
# plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
# plt.title(layer_name)
# plt.grid(False)
#plt.imshow(display_grid, aspect='auto', cmap='viridis')
# plt.show()

#step-11 save model
model.save('ildar_eye.h5') 
