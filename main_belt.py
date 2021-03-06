#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
  Object Processing
  
  Copyright 2018 Ilya Gulko <iigulko99@gmail.com>
  
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
  MA 02110-1301, USA.
  
'''

'''

Sources: 
  
  https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
  https://pythontips.com/2015/03/11/a-guide-to-finding-books-in-images-using-python-and-opencv/ 
  https://docs.opencv.org/3.4/d6/d00/tutorial_py_root.html
  
'''

import numpy as np
import cv2

def image_proccessing(image):
	'''
	this method does image tuning for further calculations.
	proccesses that are described below:
		1. bluring
		2. grayscaling
		3. thresholding
		4. edging
		5. closing edges 
	more info: https://docs.opencv.org/3.4/d6/d00/tutorial_py_root.html
	'''
	blurred = cv2.bilateralFilter(image,20,130,130)
	 
	grayscaled = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
	
	retval, thresholded = cv2.threshold(grayscaled, 175, 255, cv2.THRESH_BINARY)
	
	edged = cv2.Canny(thresholded, 10, 200)
	
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
	closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
	
	return {'blurred': blurred, 
		'graysclaled': grayscaled, 
		'thresholded': thresholded,
		'edged': edged, 
		'closed': closed }

def main():

	# reading an image as colored version
	image = cv2.imread('img/square.jpg')
		    
	# getting {dict} with all processed results
	processed = image_proccessing(image)
	
	# finding all the contours
	_,contours,hierarchy = cv2.findContours(processed['closed'], 1, 2)
	
	# defining the largest contour
	largest=-1
	nc = 0
	for i, c in enumerate(contours):
		if len(c)>nc:
			nc = len(c)
			largest = i
	
	# drawing inner contours
	cv2.drawContours(image,contours,largest,(0,0,255),3)

	# drawing out object
	x,y,w,h = cv2.boundingRect(contours[largest])
	
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),5)
	
	perimeter = cv2.arcLength(contours[largest],True)
	
	# showing text on image
	cv2.putText(image,"Aprroximate perimeter value = " + str(round(perimeter, 2)), (50, 50), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
	cv2.putText(image,"Highest point: ["+str(x)+","+str(y)+"]", (50, 85), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
	cv2.putText(image,"Lowest point: ["+str(x+w)+","+str(y+h)+"]", (50, 110), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
	
	# showing window with image; the title and path to image
	cv2.imshow('Source image', image)
	
	'''
	cv2.imshow("Closed", processed['closed'])
	cv2.imshow("Edged", processed['edged'])
	cv2.imshow('Thresholded', processed['thresholded'])
	cv2.imshow('Blurred', processed['blurred'])	
	'''
	
	cv2.waitKey(0) 
	
	cv2.destroyAllWindows()
	return 0

main()
