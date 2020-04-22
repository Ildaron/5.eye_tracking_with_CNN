# i used this soft for labelind - http://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.6.html 

import numpy as np                                                                                                                             
import glob
import pandas as pd
import PIL
from PIL import Image
import cv2

#step1 - make dataset for Y-train dataset
print ("ok2")
#df = pd.read_csv("D:/worktable/science/13.Eyes_tracking/3.My_neural_network_for_Eye/7._START_project/my_dataset_polygon_all.csv")
df = pd.read_csv("E:/0.4/3.My_neural_network_for_Eye/7._START_project/my_dataset_polygon_all1.csv")

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


