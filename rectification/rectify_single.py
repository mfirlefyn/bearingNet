'''
Rectification of catadioptric image pixels in "images" folder. Store rectified images in "rectified_images" folder.
Author: mfirlefyn
Date: 23th of February 2024
Correspondence: mfirlefyn Github Issues
Paper reference: Direct Learning of Home Vector Direction for Insect-inspired Robot Navigation
Code reference url: https://pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
'''

# import the necessary packages
import numpy as np
import argparse
import cv2

# functions
# offset coordinate at origin to middle of image, both arguments are pixel indices
def center_offset(centered,offset):
	return(centered+offset)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image, clone it for output, and then convert it to grayscale
cata_image = cv2.imread(args["image"],cv2.IMREAD_GRAYSCALE)
# rotate the image such that the front corresponds to the right (easier for traditional circle function)
cata_image = cv2.rotate(cata_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# settings
r1 = 200	# px
r2 = 400	# px
w = 1024	# px
h = 1024	# px

angle_res = 1800 	# amount of angles, thus amount of pixels in the final image

pixels = np.zeros((r2-r1+1,angle_res),dtype=np.uint8)

# runs from -180 deg up to but not including 180deg
theta = np.linspace(-np.pi,np.pi,angle_res,endpoint=False)

for radius in range(r1,r2+1):
	# rectification
	r = radius
	x_centered = r*np.cos(theta)
	y_centered = r*np.sin(theta)
	# getting the coords with offset
	x = center_offset(x_centered,w/2)
	y = center_offset(y_centered,h/2)	# the origin of opencv is left upper corner 
	# constructing mapped pixels per row
	for i in range(len(y)):
		x_coord = int(x[i])
		y_coord = int(y[i])
		pixels[r-r1][i] = cata_image[y_coord][x_coord]	# interchanged because arrays rows and cols are grid cols and rows

# store the rectified image
cv2.imwrite("rectified-eval-img.jpg",pixels)		# image was mirrored over the vertical axis

print(f"{args['image'].split('.')[0]}, {args['image'].split('.')[1]}")