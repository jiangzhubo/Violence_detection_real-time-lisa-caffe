import threading
import os
import sys
import time 

def extractRGB():
    os.system('python extractRGBframesfinal.py')
   
def extractFlow():
    os.system('python extractallopt.py')
def classification():
    os.system('python test1.py')
   
def video():
    os.system('python motion_detector.py -v'+'/home/graymatics/Downloads/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/sourcevideo/fight3.avi')

#t1 = threading.Thread(target= extractRGB)
#t2 = threading.Thread(target= extractFlow)
t3 = threading.Thread(target= classification)
t4 = threading.Thread(target= video)

#t1.start()
#t2.start()
t3.start()
time.sleep(2)
t4.start()
