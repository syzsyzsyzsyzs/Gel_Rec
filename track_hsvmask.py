import numpy as np
import cv2
import argparse
import imutils

parser =argparse.ArgumentParser()
parser.add_argument('input_img', help = 'the input image file')
args = parser.parse_args()

def nothing(x):
    pass

def colormask(filename):
    cv2.namedWindow(filename)
    pane = np.zeros((10,2000,3), np.uint8)
    cv2.namedWindow('panel', cv2.WINDOW_NORMAL)
    
    #set trackbar
    hh = 'hue high'
    hl = 'hue low'
    sh = 'saturation high'
    sl = 'saturation low'
    vh = 'value high'
    vl = 'value low'
#     mode = 'mode'

    #set ranges
    cv2.createTrackbar(hh, 'panel', 0,179, nothing)
    cv2.createTrackbar(hl, 'panel', 0,179, nothing)
    cv2.createTrackbar(sh, 'panel', 0,255, nothing)
    cv2.createTrackbar(sl, 'panel', 0,255, nothing)
    cv2.createTrackbar(vh, 'panel', 0,255, nothing)
    cv2.createTrackbar(vl, 'panel', 0,255, nothing)
#     cv2.createTrackbar(mode, filename, 0,3, nothing)

#     thv= 'th1'
#     cv2.createTrackbar(thv, filename, 127,255, nothing)

    #read img in both rgb and grayscale
    frame = cv2.imread(filename,1)
#     imgg = cv2.imread(filename,0)
    frame = imutils.resize(frame, width = 1500)
#     imgg = imutils.resize(imgg, width = 810)

    #convert rgb to hsv
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    while True:
        hul= cv2.getTrackbarPos(hl, 'panel')
        huh= cv2.getTrackbarPos(hh, 'panel')
        sal= cv2.getTrackbarPos(sl, 'panel')
        sah= cv2.getTrackbarPos(sh, 'panel')
        val= cv2.getTrackbarPos(vl, 'panel')
        vah= cv2.getTrackbarPos(vh, 'panel')
#         thva= cv2.getTrackbarPos(thv,filename)
#         modev= cv2.getTrackbarPos(mode,filename)
        
        hsvl = np.array([hul, sal, val], np.uint8)
        hsvh = np.array([huh, sah, vah], np.uint8)

        mask = cv2.inRange(hsv_img, hsvl, hsvh)
        mask = cv2.erode(mask, None, iterations = 2)
        mask = cv2.dilate(mask, None, iterations = 2)
        
#         res = cv2.bitwise_and(img, img, mask=mask)

        #set image for differnt modes
#         ret, threshold = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#         ret, img_th= cv2.threshold(imgg, thva, 255, cv2.THRESH_TOZERO
#         res2 = cv2.bitwise_and(img_th, img_th, mask=threshold)
#         res_rgb = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        #convert black to white
#         res[np.where((res==[0,0,0]).all(axis=2))] = [255,255,255]
        cv2.imshow(filename,mask)
        cv2.imshow('panel', pane)
        cv2.namedWindow('raw img')
        cv2.imshow('raw img',frame)

#         if modev ==0:
#             #show mask only
#             cv2.imshow(filename,mask)
#         elif modev ==1:
#             #show white-masked color img
#             cv2.imshow(filename,res)
#         elif modev ==2:
#             #show white-masked binary img with threshold
#             cv2.imshow(filename,threshold)
#         else:
#             #white-masked grayscale img with threshold
#             cv2.imshow(filename,res2)

        #press 'Esc' to close the window
        ch = cv2.waitKey(5)
        if ch== 27:
            break
    cv2.destroyAllWindows()

    return mask

colormask(args.input_img)
