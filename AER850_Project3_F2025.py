import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO


if __name__ == "__main__":
    
    img = cv.imread('data/motherboard_image.JPEG')
    imgg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgg = cv.medianBlur(imgg, 5)
    ret,thresh = cv.threshold(imgg,90,170,cv.THRESH_BINARY_INV)
    #thresh = cv.adaptiveThreshold(imgg, 500, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 111, 3)
    
    cv.namedWindow('Image', cv.WINDOW_NORMAL)
    cv.imshow('Image', thresh)
    cv.waitKey(0)
    #cv.imwrite('imgg_thresh.jpg', thresh)
    cv.destroyAllWindows()
    
    contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
                                          
    # draw contours on the original image
    
    
    mask = np.zeros_like(img)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        cv.drawContours(mask, [largest_contour], -1, (255, 255, 255), -1)    
        
    masked_image = cv.bitwise_and(img, mask)
    
    # cv.namedWindow('Original Image', cv.WINDOW_NORMAL)
    # cv.namedWindow('Mask', cv.WINDOW_NORMAL)
    cv.namedWindow('Masked Image', cv.WINDOW_NORMAL)
    # cv.imshow('Original Image', img)
    # cv.imshow('Mask', mask)
    cv.imshow('Masked Image', masked_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
    
    model = YOLO("yolo11n.pt")
   
    trained = model.train(data="data/data/data/data.yaml", batch = 8, epochs = 100, imgsz = 900, name = "Prj3Mdl")
    
