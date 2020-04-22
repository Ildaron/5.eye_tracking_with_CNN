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

#input data create a cicle
b=np.array([])
size_image=50
radius=4
dataset_amount = 50

dataset_x=np.zeros((dataset_amount, size_image, size_image))
dataset_y=np.zeros((dataset_amount, radius, radius))


#start calculation
imgs=np.zeros((1,size_image,size_image))
graph_resolution=size_image-1

radius_data=[]
radius_data_pos=[]

for y in range (-radius,radius+1,1):        
 x=math.sqrt(radius*radius-y*y)  # create_dataset - create the cicle
 negative_x=x
 negative_x*= -1
 
 radius_data.append(round(x))
 radius_data.append(y)

 radius_data.append(round(negative_x))
 radius_data.append(y)

for image in range (dataset_amount):
 circle_center=random.randint(radius, size_image-radius-1)   # create_dataset - random move the center of coordinates
 for a in (radius_data):
  radius_data_pos.append(a+circle_center)

   #- this data for train model - it is Y 
 lenght=len(radius_data)

 imgs=np.zeros((1,size_image,size_image))
 for a in range(lenght):
  if not a % 2: 
   imgs[0,radius_data_pos[a+1+(len(radius_data_pos)-36)],radius_data_pos[a+len(radius_data_pos)-36]]=1  # create_dataset - move the center of coordinates of a cicle to the positive axis
   
# print (imgs) - - this data for train model - it is X
 b=np.append(b,imgs)
b=b.reshape(dataset_amount,size_image,size_image)
dataset_X=b
#print (radius_data_pos)
dataset_Y=np.array(radius_data_pos)
#print (len(datasex_Y))
#print ("ok")
dataset_Y=dataset_Y.reshape(image+1,radius+2,radius+2)
#print (datasex_Y)
# to see created data 

#plt.imshow(datasex_Y[1].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, graph_resolution, 0, graph_resolution])
#plt.show()
#plt.imshow(datasex_X[1].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, graph_resolution, 0, graph_resolution])
#plt.show()
#print (dataset_Y.shape)
#print (dataset_X.shape)

type_x=dataset_X.shape
X = dataset_X.reshape(type_x[0],type_x[1],type_x[2], 1) # 
#print (X.shape)
#type_y=dataset_Y.shape
#Y = dataset_Y.reshape(type_y[0],type_y[1],type_y[2], 1) # 
#print (Y.shape)
Y=dataset_Y.reshape(dataset_amount,36)

i = int(0.8 * dataset_amount)
train_X = X[:i]
test_X = X[i:]
train_y = Y[:i]
test_y = Y[i:]

#test_imgs = dataset_amount[i:]
#test_bboxes = bboxes[i:]

#print ("model_start")
model = models.Sequential()
#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size_image, size_image, 1)))
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)))  #  img_size ????????????? не должно ли стать менье картинка изображения?
model.add(layers.Conv2D(32, (3, 3), activation='relu'))  
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))  
model.add(layers.Flatten())
model.add(layers.Dense(600, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
#model.add(layers.Dense(y.shape[-1], activation='sigmoid'))
#model.add(layers.Dense(y.shape[-1]))
model.add(layers.Dense(36))
#print ("model_finish")
model.summary()
model.compile('adadelta', 'mse')

#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
#from keras import optimizers
#model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),  metrics=['acc'])

#print ("train_X", train_X.shape)
#print ("train_y", train_y.shape)
model.fit(train_X, train_y, nb_epoch=10000, validation_data=(test_X, test_y), verbose=2)
print ("ok3")
pred_y = model.predict(test_X)

#print (pred_y[0,1])

# to see Y predict start
pred_y_list=[]
for a in range(0,36,1):
 print (a)
 pred_y_list.append(int(pred_y[0,a]))
check_y=np.zeros((1,size_image,size_image))
pred_y=pred_y_list
print (pred_y)
lenght=len(radius_data)

for a in range(lenght):
 if not a % 2:     
  check_y[0,round(pred_y[a+1]),round(pred_y[a])]=1

plt.imshow(check_y[0].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, size_image, 0, size_image])
plt.show()

# to see Y predict stop


# to See X start

print(test_X.shape)
z=test_X[0]
print (z.shape)
z=z.reshape(size_image,size_image)
print (z.shape)
print (z)
plt.imshow(z, cmap='Greys', interpolation='none', origin='lower', extent=[0, 9, 0, 9])
plt.show()

# to See X stop
