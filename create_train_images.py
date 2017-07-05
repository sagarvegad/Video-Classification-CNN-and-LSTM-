import numpy as np
import os
import pandas as pd
import cv2
from scipy.misc import imread,imresize
import pickle

parent = os.listdir("/Users/.../video/images")

x = []
y = []
count = 0
output = 0

for video_class in parent[1:]:
    print video_class
    child = os.listdir("/Users/.../video/images" + "/" + video_class)
    for class_i in child[1:]:
        sub_child = os.listdir("/Users/.../video/images" + "/" + video_class + "/" + class_i)
        for image_fol in sub_child[1:]:
            if (video_class ==  'class_4' ):
                if(count%4 == 0):
                    image = imread("/Users/.../video/images" + "/" + video_class + "/" + class_i + "/" + image_fol)
                    image = imresize(image , (224,224))
                    x.append(image)
                    y.append(output)
                    cv2.imwrite('/Users/.../video/' + video_class + '/' + str(count) + '_' + image_fol,image)
                count+=1
            
            else:
                if(count%8 == 0):
                    image = imread("/Users/.../video/images" + "/" + video_class + "/" + class_i + "/" + image_fol)
                    image = imresize(image , (224,224))
                    x.append(image)
                    y.append(output)
                    cv2.imwrite('/Users/.../video/' + video_class + '/' + str(count) + '_' + image_fol,image)
                count+=1
    output+=1
x = np.array(x)
y = np.array(y)
print("x",len(x),"y",len(y))


