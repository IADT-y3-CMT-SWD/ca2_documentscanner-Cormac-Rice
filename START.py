import cv2 as cv
import numpy as np

#Trackbar Code
def nothing(x):
    pass
def initializeTrackbars(intialTracbarVal=125):
    cv.namedWindow("WinTrackbars")
    cv.resizeWindow("WinTrackbars", 360, 240)
    cv.createTrackbar("Threshold1", "WinTrackbars", intialTracbarVal,255, nothing)
    cv.createTrackbar("Threshold2", "WinTrackbars", intialTracbarVal, 255, nothing)
 
initializeTrackbars()

def valTrackbars():
    Threshold1 = cv.getTrackbarPos("Threshold1", "WinTrackbars")
    Threshold2 = cv.getTrackbarPos("Threshold2", "WinTrackbars")
    src = Threshold1,Threshold2
    return src


#Finds the biggest countour
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv.contourArea(i)
        #Sets the maximum area size to 4999
        if area > 5000:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area
 
#Reorders the points to warp image
def reorder(thePoints):
    thePoints = thePoints.reshape((4, 2))
    thePoints = np.zeros((4, 1, 2), dtype=np.int32)
    add = thePoints.sum(1)
 
    theNewPoints[0] = thePoints[np.argmin(add)]
    theNewPoints[3] = thePoints[np.argmax(add)]
    diff = np.diff(thePoints, axis=1)
    theNewPoints[1] = thePoints[np.argmin(diff)]
    theNewPoints[2] = thePoints[np.argmax(diff)]
    
    return    
 

#If there is no webcam feed available
webCamFeed = False
pathImage = "Images\\image004.jpg"
#If there is webcam feed availaable
cap = cv.VideoCapture(0)
cap.set(10, 160)
heightImg = 500
widthImg = 600

count = 0

while True:
    #If there is webcam feed
    if webCamFeed:
        success, img = cap.read()
    #If there is no webcam feed    
    else:
        img = cv.imread(pathImage)
    #Resizing the image
    thres = valTrackbars()
    img = cv.resize(img, (heightImg, widthImg))
    #Creates a blank image
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    #Turns the image into Greyscale
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #Adds Gaussian Blur to the image 
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
    # thres = valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgCanny = cv.Canny(imgBlur, thres[0], thres[1]) #thres[0], thres[1])  # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    #Dilates the image
    imgDial = cv.dilate(imgCanny, kernel, iterations=2)
    #Erodes the image
    imgThreshold = cv.erode(imgDial, kernel, iterations=1)
    imgContours = img.copy()
    contours, hierarchy = cv.findContours(imgThreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(imgThreshold, contours, -1, (255, 0, 0), 10)
    biggest, max_area = biggestContour(contours)
    theNewPoints = reorder(biggest)
