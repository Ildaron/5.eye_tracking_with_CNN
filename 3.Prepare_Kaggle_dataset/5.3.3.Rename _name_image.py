import os
folder_name="D:/test/"
image_start=0

for filename in os.listdir(folder_name):
 image_start=image_start+1
 dst = str(image_start) + ".jpg"
 src = folder_name+ filename
 dst = folder_name+ dst        
 os.rename(src, dst)

