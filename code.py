import cv2
import numpy as np

f1 = open('points1x.txt','w')
f1.close()

f2 = open('points1y.txt','w')
f2.close()

f3 = open('points2x.txt','w')
f3.close()

f4 = open('points2y.txt','w')
f4.close() 


def edge_detect(image):
    # edge detection

    gradients_sobelx = cv2.Sobel(image,-1,1,0)
    gradients_sobely = cv2.Sobel(image,-1,0,1)
    gradients_sobelxy = cv2.addWeighted(gradients_sobelx,0.5,gradients_sobely,0.5,0)

    gradients_laplacian = cv2.Laplacian(image,-1)

    canny_output = cv2.Canny(image,100,150)

    #cv2.imshow('Sobel x',gradients_sobelx)
    #cv2.imshow('Sobel y',gradients_sobely)
    #cv2.imshow('Sobel x+y',gradients_sobelxy)
    #cv2.imshow('laplacian',gradients_laplacian)
    cv2.imshow('Canny',canny_output)
    # cv2.imwrite("/home/aayush/Downloads/Opencv-P/images/fig2.png",canny_output)

def mousePoints(event,x,y,flags,params):

    # Mouse event detection

    font = cv2.FONT_HERSHEY_SIMPLEX
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(imw,(x,y),10,(0,255,0))

        print(x,y)

        xc.append(x)
        f1 = open("points1x.txt",'a')
        f1.write(str(x)+ " ")
        f1.close()

        yc.append(y)
        f2 = open("points1y.txt",'a')
        f2.write(str(y)+ " ")
        f2.close()

        cv2.imshow('image',imw)


def mousePoints2(event,x,y,flags,params):

    # Mouse event detection

    font = cv2.FONT_HERSHEY_SIMPLEX
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(im2,(x,y),10,(255,0,0))

        print(x,y)

        x2c.append(x)
        f3 = open("points2x.txt",'a')
        f3.write(str(x)+ " ")
        f3.close()

        y2c.append(y)
        f4 = open("points2y.txt",'a')
        f4.write(str(y)+ " ")
        f4.close()

        cv2.imshow('image2',im2)

xc = list()
yc = list()
x2c = []
y2c = []

image = cv2.imread('images/star.png')
image = cv2.resize(image,None,fx=0.8,fy = 0.8)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(binary, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)

im_copy = image.copy()
im_copy = cv2.drawContours(im_copy,contours,-1,(0,255,0),thickness=2)

# cv2.imshow("Grayscale",gray)
# cv2.imshow("Drawn",im_copy)
# cv2.imshow("Original",image)
cv2.imshow("Binary",binary)
# cv2.imwrite("/home/aayush/Downloads/Opencv-P/images/fig1.png",binary)

#print (contours)

# edge_detect(binary)

image2 = cv2.imread('images/circle.png')
image2 = cv2.resize(image2,None,fx=0.8,fy = 0.8)

gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

_, binary2 = cv2.threshold(gray2,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours2, hierarchy2 = cv2.findContours(binary2, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)


# cv2.imshow("Grayscale",gray)
# cv2.imshow("Drawn",im_copy)
# cv2.imshow("Original",image)
cv2.imshow("Binary2",binary2)
# cv2.imwrite("/home/aayush/Downloads/Opencv-P/images/fig2.png",binary)


# edge_detect(binary2)

imw = cv2.imread('images/fig1.png')
cv2.imshow('image',imw)
cv2.setMouseCallback("image",mousePoints)

im2 = cv2.imread('images/fig2.png')
cv2.imshow('image2',im2)
cv2.setMouseCallback("image2",mousePoints2)

# img_concat = np.concatenate((image,image2),axis = 1)
# cv2.imshow('concatenated',img_concat)
    
cv2.waitKey(0)
cv2.destroyAllWindows()

