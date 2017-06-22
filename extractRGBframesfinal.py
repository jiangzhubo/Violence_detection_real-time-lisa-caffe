#! encoding: UTF-8
import os
import cv2
import PIL
from PIL import Image
videos_src_path = '/home/graymatics/Downloads/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/sourcevideo/'
videos_save_path = '/home/graymatics/Downloads/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/testimage/'
videos = os.listdir(videos_src_path)
videos = filter(lambda x: x.endswith('avi'), videos)
#change the type of video .
for each_video in videos:
    sec = 0
    tam =0
    h =0
    each_video_name, _ = each_video.split('.')
    #each_video_name = each_video
    print each_video_name
    each_video_full_patha = os.path.join(videos_src_path, each_video)
    each_video_full_path = os.path.join(videos_src_path, each_video_name)
    cap= cv2.VideoCapture(each_video_full_patha)
    os.mkdir(videos_save_path+ '/' +each_video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    each_video_save_full_path_5s = os.path.join(videos_save_path, each_video_name) + str(sec)+'/'
    #sec = sec+1
    frame_count = 0 # the  frame which starts collecting 
    mir = 4 # spilt video into 1 sec by 1 sec
    timeF=1 #split video into 1 frame by frame
    success = True
    while(success):
	success, frame = cap.read()
	params = []
	params.append(cv2.IMWRITE_PXM_BINARY)
	params.append(1)
	if (frame_count%int(fps*mir) ==0 ):
                h= h+1 
		os.mkdir(videos_save_path+ '/' +"%d"%h)               
	if(frame_count%timeF == 0 ):
		#cv2.imwrite(videos_save_path+ '/' +each_video_name + '/' + each_video_name+".%04d.jpg" % frame_count,frame)
              if success == True:
                  cv2.imwrite(videos_save_path+ '/' +"%d"%h+'/'+"%d"%h+".%04d.jpg" % frame_count,frame)
	frame_count = frame_count + 1
	if (frame_count  == int(fps*mir)):
		frame_count =0



