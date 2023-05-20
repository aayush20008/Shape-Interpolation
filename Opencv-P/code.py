import cv2
import numpy

image = cv2.imread('images/star.png')
image = cv2.resize(image,None,fx=0.8,fy = 0.8)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(binary, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)

im_copy = image.copy()
im_copy = cv2.drawContours(im_copy,contours,-1,(0,255,0),thickness=2)

cv2.imshow("Grayscale",gray)
cv2.imshow("Drawn",im_copy)
# cv2.imshow("Original",image)
cv2.imshow("Binary",binary)

print (contours)

# im_copy1 = image.copy()
contours, hierarchy = cv2.findContours(binary, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
# im_copy1 = cv2.drawContours(im_copy1,contours,-1,(0,255,0),thickness=2)

# cv2.imshow("New",im_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()

