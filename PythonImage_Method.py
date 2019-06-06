
# coding: utf-8

# In[1]:


# import packages
from collections import deque # list-like data structure, for contrail of object
import numpy as np
import argparse
import imutils
import cv2
import csv
import os


# In[ ]:


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', # '--video' video file path
               help = 'path to the video file')
ap.add_argument('-b', '--buffer', type = int, default = 64, # '--buffer' -> max size of deque -> length of contrail
               help = 'max buffer size')
args = vars(ap.parse_args())


# Circle mask
img_for_mask = cv2.imread('mask.png', 0)
ret, thres1 = cv2.threshold(img_for_mask,50,255,cv2.THRESH_BINARY)
circle_img = thres1

# define the lower & upper boundaries of the 'green'
# ball in the HSV color space, then initialize the list of tracked points
black = (0,40,100)
redLower = (10, 220, 255)
redUpper = (130, 40, 100)
white = (180,220,255)


blueLower = (55, 20, 20)
blueUpper = (135, 255, 240)

pts_red = deque(maxlen = args['buffer']) # Initialize the deque & pts
pts_blue = deque(maxlen = args['buffer']) 

# if a video path was not supplied, grab the ref to the webcam
if not args.get('video', False):
    camera = cv2.VideoCapture(0)
    
# otherwise, grab a ref to the video file
else:
    camera = cv2.VideoCapture(args['video']) # grab a ref    

timer = 0
x = 0
if os.path.exists('coord_red.csv'):
    os.remove('coord_red.csv')

if os.path.exists('coord_blue.csv'):
    os.remove('coord_blue.csv')
    
# Keep looping
while True:
    # grab the current frame
    (grabbed , frame_raw) = camera.read() # grabbed -> whether the frame was successfully read or not; frame -> video itself

    
    # if we are viewing a video and we did not grab a frame, then we have reached the end of the video
    if args.get('video') and not grabbed:
        break
    timer += (1/60)
    minute = x//3600
    second = (x%3600)//60
    frame = cv2.bitwise_and(frame_raw, frame_raw, mask  = circle_img)
    
    
    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width = 1500)
    
    # blurblue = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small blobs left in the mask
    mask_red1 = cv2.inRange(hsv, black, redLower)
    mask_red2 = cv2.inRange(hsv, redUpper, white)
    mask_red = cv2.bitwise_or(mask_red1,mask_red2)

    # mask_red = cv2.inRange(hsv, redLower, redUpper) # give a binary mask
    mask_red = cv2.erode(mask_red, None, iterations = 2) # remove small blobs on mask
    mask_red = cv2.dilate(mask_red, None, iterations = 2)
    
    mask_blue = cv2.inRange(hsv, blueLower, blueUpper) # give a binary mask
    mask_blue = cv2.erode(mask_blue, None, iterations = 2) # remove small blobs on mask
    mask_blue = cv2.dilate(mask_blue, None, iterations = 2)
    
    # find contours in the mask and initialize the current (x,y) center of the ball
    cnts_red = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts_blue = cv2.findContours(mask_blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center_red = None
    center_blue = None
    
    # only procced if at least one contour was found
    if len(cnts_red) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c_red = max(cnts_red, key = cv2.contourArea)
        ((x_red, y_red), radius_red) = cv2.minEnclosingCircle(c_red)
        M_red = cv2.moments(c_red)
        center_red = (int(M_red['m10'] / M_red['m00']), int(M_red['m01'] / M_red['m00']))
                
        # only proceed if the radius meets a minimum size
        if radius_red > 4 and radius_red < 18:
            # draw the circle and centroid on the frame, then update the list of tracked points
            cv2.circle(frame, (int(x_red), int(y_red)), int(radius_red), (0, 255, 255), 2)
            # cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        
            coord_red = '%f, %f, %f\n' % (x_red, y_red, timer)
            with open('coord_red.txt', 'a') as red:
                red.write(coord_red)
                
    if len(cnts_blue) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c_blue = max(cnts_blue, key = cv2.contourArea)
        ((x_blue, y_blue), radius_blue) = cv2.minEnclosingCircle(c_blue)
        M_blue = cv2.moments(c_blue)
        center_blue = (int(M_blue['m10'] / M_blue['m00']), int(M_blue['m01'] / M_blue['m00']))
                
        # only proceed if the radius meets a minimum size
        if radius_blue > 3 and radius_blue < 12:
            # draw the circle and centroid on the frame, then update the list of tracked points
            cv2.circle(frame, (int(x_blue), int(y_blue)), int(radius_blue), (255, 255, 0), 2)
            # cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
            coord_blue = '%f, %f, %f\n' % (x_blue, y_blue, timer)
            with open('coord_blue.txt', 'a') as blue:
                blue.write(coord_blue)
        
        
        # update the points queue
        pts_red.appendleft(center_red)
        pts_blue.appendleft(center_blue)
        
        # draw contrail
        # loop over the set of tracked points
        for i in range(1, len(pts_red)):
            # if either of the tracked points are None, ignore them
            if pts_red[i - 1] is None or pts_red[i] is None:
                continue
            
            # otherwise, compute the thickness of the line and draw the connecting lines
            thickness = int(np.sqrt(args['buffer'] / float(i + 1)) * 0.5)
            cv2.line(frame, pts_red[i - 1], pts_red[i], (0, 0, 255), thickness)
        
        for i in range(1, len(pts_blue)):
            # if either of the tracked points are None, ignore them
            if pts_blue[i - 1] is None or pts_blue[i] is None:
                continue
            
            # otherwise, compute the thickness of the line and draw the connecting lines
            thickness = int(np.sqrt(args['buffer'] / float(i + 1)) * 0.5)
            cv2.line(frame, pts_blue[i - 1], pts_blue[i], (255, 0, 0), thickness)
    
    clock = '%d min  %d sec' % (minute, second)
    x += 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, clock, (1100, 700), font, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.circle(frame, (1100, 600), 5, (0, 0, 255), 1)
    cv2.circle(frame, (1100, 600), 10, (0, 255, 255), 1)
    cv2.circle(frame, (1100, 600), 15, (0, 255, 0), 1)
    cv2.circle(frame, (1100, 600), 20, (255, 255, 0), 1)
    cv2.circle(frame, (1100, 600), 25, (255, 0, 0), 1)
    cv2.putText(frame, '5, 10, 15, 20, 25', (1100, 500), font, 1, (255,255,255), 2, cv2.LINE_AA)
    
    # show the frame to our screen
    #cv2.imshow('mask_red', mask_red)
    #cv2.imshow('mask_blue', mask_blue)
    cv2.imshow('Frame', frame)
    cv2.imshow('mask_red', mask_red)
    cv2.imshow('mask_blue', mask_blue)
    key = cv2.waitKey(1) & 0xff

    # if the 'q' key is pressed, stop the loop
    if key == ord('q'):
        break
            
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

with open ('coord_red.csv', 'w') as csvfile:
    temp = csv.writer(csvfile, dialect = 'excel')
    with open('coord_red.txt', 'r') as filein:
        for line in filein:
            line_list = line.strip('\n').split(',')
            temp.writerow(line_list)

with open ('coord_blue.csv', 'w') as csvfile:
    temp = csv.writer(csvfile, dialect = 'excel')
    with open('coord_blue.txt', 'r') as filein:
        for line in filein:
            line_list = line.strip('\n').split(',')
            temp.writerow(line_list)
os.remove('coord_red.txt')
os.remove('coord_blue.txt')
