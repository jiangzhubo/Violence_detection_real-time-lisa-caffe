# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None
fps = camera.get(cv2.CAP_PROP_FPS)
# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()
	

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	#if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	(_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	count = 0
 	frame_count =0
        text = 'normal activity'
	# loop over the contou
        #print ('cnts=%s'%(cnts), len(cnts))
	frame_count = frame_count + 1
	for c in cnts:
                #print('c=%s'%(c),len(cnts))
		# if the contour is too small, ignore it
		#if cv2.contourArea(c) < args["min_area"]:
			#continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		#(x, y, w, h) = cv2.boundingRect(c)
		count += 1
		#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
              
                text_file = open('VideoOutput.txt','r')             
	       	text1 = text_file.read()
	        #print(text)
                text = text1.split('\n')
                #print(text[0])
                #print(text[1])
	        text_file.close()
	        frame_count =0
	#print('number of boxes: {}').format(count)
	# draw the text and timestamp on the frame
	cv2.putText(frame, text[0], (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, text[1], (10, 60),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 120), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
        time.sleep(0.05)
        frame = cv2.resize(frame,(1200,980))
	cv2.imshow("Violence Detector", frame)
	cv2.namedWindow('Violence Detector',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Violence Detector',1200,980)
        key = cv2.waitKey(1) & 0xFF
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
