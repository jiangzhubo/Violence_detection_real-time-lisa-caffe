import cv2
import numpy as np
import os
import threading
import time
import sys
import classify_LRCN
video = '/home/graymatics/Downloads/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/video/fighting2.avi'
add1 = '/home/graymatics/Downloads/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/testimage/'
add2 = '/home/graymatics/Downloads/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/testflow/'
text = 'Normal activity'


def classify(video,add1,add2):
    result = classify_LRCN.test40(video,add1,add2)
    return result
acti=['BasketballDunk','BodyWeightSquats','Bowling','BoxingPunchingBag','BoxingSpeedBag','Diving',
'Fencing',
'GolfSwing',
'Hammering',
'HammerThrow',
'PullUps',
'Punch',
'PushUps',
'Shotput',
'SumoWrestling',
'groupfighting',
'WallPushups',
'ThrowDiscus',
'Swing',
'MoppingFloor',
'Archery']

time.sleep(5)
num = 0.9
def text1(h,add1,add2):
	violence = classify(h,add1,add2)
        #return(violence)	
	if violence in acti:
	    return ('Caution! violent activity detected!')
	if violence not in acti:
	    return('Normal activity')

for h in range(10):
      text = text1(h,add1,add2)
      text_file = open("VideoOutput.txt", "w")
      #print(yhtext)	
      text_file.write("%s \n" %(text))
      text_file.close()    



