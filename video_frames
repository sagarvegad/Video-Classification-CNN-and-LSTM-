import os
import math
import cv2

listing = os.listdir("/Users/svdj16/Documents/6_sem_mini_project/video/class_3")
count = 1
for file in listing:
    video = cv2.VideoCapture("/Users/svdj16/Documents/6_sem_mini_project/video/class_3/" + file)
    print(video.isOpened())
    framerate = video.get(5)
    os.makedirs("/Users/svdj16/Documents/6_sem_mini_project/video/class_3_images/" + "video_" + str(int(count)))
    while (video.isOpened()):
        frameId = video.get(1)
        success,image = video.read()
        if( image != None ):
            image=cv2.resize(image,(224,224), interpolation = cv2.INTER_AREA)
        if (success != True):
            break
        if (frameId % math.floor(framerate) == 0):
            filename = "/Users/svdj16/Documents/6_sem_mini_project/video/class_3_images/video_" + str(int(count)) + "/image_" + str(int(frameId / math.floor(framerate))+1) + ".jpg"
            print(filename)
            cv2.imwrite(filename,image)
    video.release()
    print('done')
    count+=1
