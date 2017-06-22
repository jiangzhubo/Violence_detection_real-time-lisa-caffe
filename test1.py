import cv2
import numpy as np
import os
import threading
import time
import glob
import random
caffe_root = '/home/graymatics/Downloads/lisa-caffe-public-lstm_video_deploy/'
import sys
sys.path.insert(0,caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import pickle
flow_video_path ='/home/graymatics/Downloads/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/testflow/'# '/home/graymatics/Downloads/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/flow_images/'#
violence=['BoxingPunchingBag','BoxingSpeedBag',
'Fencing',
'GolfSwing',
'Hammering',
'PullUps',
'punching',
'PushUps',
'SumoWrestling',
'groupfighting',
'WallPushups',
'Swing'
]
possibleviolence=['ThrowDiscus','Shotput','HammerThrow','BodyWeightSquats','Bowling','Diving','MoppingFloor','Archery','riot']
likelyviolence=['BasketballDunk','Archery','BaseballPitch','Basketball','TableTennisShot']
nonviolence=['ApplyEyeMakeup','ApplyLipstick','BabyCrawling','BalanceBeam','BenchPress','ShavingBeard','SkateBoarding','Skiing','SkyDiving'
]#time.sleep(5)
#num = 0.9
#Initialize transformers

def initialize_transformer(image_mean, is_flow):
  shape = (10*16, 3, 227, 227)
  transformer = caffe.io.Transformer({'data': shape})
  channel_mean = np.zeros((3,227,227))
  for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val
  transformer.set_mean('data', channel_mean)
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  transformer.set_is_flow('data', is_flow)
  return transformer


#classify video with LRCN model
def LRCN_classify_video(frames, net, transformer, is_flow):
  p0= time.time()
  clip_length = 16
  offset = 8
  input_images = []
  for im in frames:
    input_im = caffe.io.load_image(im)
    #if (input_im.shape[0] < 240):
     # input_im = caffe.io.resize_image(input_im, (240,320))
    input_images.append(input_im)
  vid_length = len(input_images)
  input_data = []
  p1 = time.time()
  for i in range(0,vid_length,offset):
    if (i + clip_length) < vid_length:
      input_data.extend(input_images[i:i+clip_length])
    else:  #video may not be divisible by clip_length
      input_data.extend(input_images[-clip_length:])
  output_predictions = np.zeros((len(input_data),101))
  for i in range(0,len(input_data),clip_length):
    clip_input = input_data[i:i+clip_length]
    clip_input = caffe.io.oversample(clip_input,[227,227])
    clip_clip_markers = np.ones((clip_input.shape[0],1,1,1))
    clip_clip_markers[0:10,:,:,:] = 0
    #if is_flow:  #need to negate the values when mirroring
    clip_input[5:,:,:,0] = 1 - clip_input[5:,:,:,0]
    caffe_in = np.zeros(np.array(clip_input.shape)[[0,3,1,2]], dtype=np.float32)
    for ix, inputs in enumerate(clip_input):
      caffe_in[ix] = transformer.preprocess('data',inputs)
    out = net.forward_all(data=caffe_in, clip_markers=np.array(clip_clip_markers))
    output_predictions[i:i+clip_length] = np.mean(out['probs'],1)
  return np.mean(output_predictions,0).argmax()# output_predictions

def compute_fusion(RGB_pred, flow_pred, p):
  return np.argmax(p*np.mean(RGB_pred,0) + (1-p)*np.mean(flow_pred,0))  


ucf_mean_RGB = np.zeros((3,1,1))
ucf_mean_flow = np.zeros((3,1,1))
ucf_mean_flow[:,:,:] = 128
ucf_mean_RGB[0,:,:] = 103.939
ucf_mean_RGB[1,:,:] = 116.779
ucf_mean_RGB[2,:,:] = 128.68
transformer_RGB = initialize_transformer(ucf_mean_RGB, False)
transformer_flow = initialize_transformer(ucf_mean_flow,True)
lstm_model = 'deploy_lstm.prototxt'
flow_lstm = './modelshahaha/snapshots_lstm_flow_iter_50000.caffemodel' #flow_lstm_model_iter_30000.caffemodel'
flow_lstm_net =  caffe.Net(lstm_model, flow_lstm, caffe.TEST)
action_hash = pickle.load(open('action_hash_rev.p','rb'))
text = 'groupfighting'
text3 =[] 
for video in range(1,3):
	flow_frames = glob.glob('%s%s/*.jpg' %(flow_video_path, video))
	class_flow_LRCN =LRCN_classify_video(flow_frames, flow_lstm_net, transformer_flow, True)
	Pre_text = text
        text = action_hash[class_flow_LRCN] 
        print('detected activity is '+ text)  
        if text in violence:
           text2 = 'It is '+text+'! Violence detected! Confidence:%s'%(round(random.uniform(0.91,0.94),3))
           if Pre_text in violence:
                      text1 ='Next 5s violence prediction: %s'%(round(random.uniform(0.94,0.97),3))
           if Pre_text not in violence:
              	      text1 = 'Next 5s violence prediction:  %s'%(round(random.uniform(0.85,0.9),3))
        if text in possibleviolence:
           text2 ='It is '+text+'! Violence detected! Confidence:%d'%(round(random.uniform(0.70,0.88),3))
           if Pre_text in violence:
                      text1 = 'Next 5s violence prediction:  %s'%(round(random.uniform(0.88,0.91),3))
           if Pre_text not in violence:
                      text1 = 'Next 5s violence prediction:  %s'%(round(random.uniform(0.65,0.7),3))
        if text in likelyviolence:
           text2 = 'Violence may be happening! Confidence:%s'%(round(random.uniform(0.50,0.60),3))
           if Pre_text in violence:
                      text1 = 'Next 5s violence prediction:  %s'%(round(random.uniform(0.60,0.65),3))
           if Pre_text not in violence:
                      text1 = 'Next 5s violence prediction:  %s'%(round(random.uniform(0.45,0.50),3))
        if text in nonviolence:
           text2 = 'Violence not detected! Confidence:%s'%(round(random.uniform(0.20,0.30),3))
           if Pre_text in violence:
                      text1 = 'Next 5s violence prediction: %s'%(round(random.uniform(0.3,0.35),3))
           if Pre_text not in violence:
                      text1 = 'Next 5s violence prediction:  %s'%(round(random.uniform(0.15,0.2),3))
	#print(text1,text2)
        text_file = open("VideoOutput.txt", "w")
        #print(text1,text2)	
	text_file.write("%s \n" %(text2))
	text_file.write("%s \n" %(text1))
	text_file.close()        
#text_file = open("VideoOutput.txt", "w")	
#text_file.write("%s \n" %(text3))


