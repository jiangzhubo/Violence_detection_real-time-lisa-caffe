import numpy as np
import cv2
import video
import os
import subprocess

#!/usr/bin/env python


# Python 2/3 compatibility


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def main(fn,save_add):
	if __name__ == '__main__':
	    import sys
	    videos = os.listdir(fn)
	    videos = filter(lambda x: x.endswith('avi'), videos) #change the type of video .
	    for each_video in videos:
                    each_video_name,_ = each_video.split('.')
		    pathe = fn +'/' + each_video
		    cam = video.create_capture(pathe)
		    #os.mkdir(save_add + '/' +  each_video_name)
                    #length = cam.get(cv2.CAP_PROP_FRAME_COUNT)
		    ret, prev = cam.read()
		    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                    fps = cam.get(cv2.CAP_PROP_FPS)
		    cur_glitch = prev.copy()
		    sec =0
                    mir = 3
                    h =1
                    os.mkdir(save_add+ '/' +"%d"%h)
		    fm =save_add 
		   #print(fm)
		    while ret:
			ret, img = cam.read()
		        if ret > 0 :
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
				prevgray = gray
				sec =sec+1 
                                #flow = cv2.pyrDown(flow)
			        bgr = draw_hsv(flow)
                                if (sec%int(fps*mir) ==0 ):
                                    h = h+1
                                    sec = 0 
		                    os.mkdir(save_add+ '/' +"%d"%h)
                                #bgr = cv2.pyrDown(bgr)
                                #bgr = cv2.pyrDown(bgr)
		                bgr = cv2.resize(bgr,(320,240))
                                #bgr = cv2.pyrDown(bgr)
                                #bgr = cv2.fastNlMeansDenoisingColored(trg,None,10,10,7,21) #reduce the noise   
                                if( sec%2==0):
			        	cv2.imwrite(fm +'/'+"%d"%h  +'/'+"%d"%h+'.%04d.jpg'%sec,bgr,[cv2.IMWRITE_JPEG_QUALITY, 12])
                                print(fm +"%d"%h  +'/'+"%d"%h+'.%04d.jpg'%sec)
					
				ch = cv2.waitKey(5)
				if ch == 27:
				    break
	    cam.release()



videos_src_path_O = '/home/graymatics/Downloads/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/sourcevideo/'
videos_save_path_O = '/home/graymatics/Downloads/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/testflow/'
main(videos_src_path_O,videos_save_path_O)
# extract_optical_flow
