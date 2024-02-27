'''
Rectification of catadioptric image pixels in "images" folder. Store rectified images in "rectified_images" folder.
Author: mfirlefyn
Date: 23th of February 2024
Correspondence: mfirlefyn Github Issues
Paper reference: Direct Learning of Home Vector Direction for Insect-inspired Robot Navigation
Code reference url: https://pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
'''

# import the necessary packages
from scipy import ndimage
import numpy as np
import cv2

# functions
# offset coordinate at origin to middle of image, both arguments are pixel indices
def center_offset(centered,offset):
	return(centered+offset)

# settings
am_imgs = 100
am_circ_deg = 360

r1 = 200	# px
r2 = 400	# px
w = 1024	# px
h = 1024	# px

angle_res = 1800 	# amount of pixels in the final image, must be divisible by 360 angles (full circle)

for idx in range(am_imgs):
	# load the image and then convert it to grayscale
	cata_image_w = cv2.imread("images/omni-dir-img-"+str(idx)+".jpg",cv2.IMREAD_GRAYSCALE)		# west
	# rotate the image such that the front corresponds to the right, reference frame transform (easier for traditional circle function)
	cata_image_n = ndimage.rotate(cata_image_w, 90) #cv2.rotate(cata_image_w, cv2.ROTATE_90_COUNTERCLOCKWISE)		# north	
	# initialize pixel arrays to hold rectifications
	pixels_n = np.zeros((r2-r1+1,angle_res),dtype=np.uint8)	# rectification with Northern gaze
	pixels = np.zeros((r2-r1+1,angle_res),dtype=np.uint8)	# placeholder for all other gaze directions

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
			pixels_n[r-r1][i] = cata_image_n[y_coord][x_coord]	# interchanged because arrays rows and cols are grid cols and rows

	# first store the rectified image with Northern gaze
	print("Rectify: Constructing rectified image " + str(idx))
	cv2.imwrite("rectified_images/rectified-img-"+str(idx)+".jpg",pixels_n)
	
	# construct and store other gaze angles from Northern reference
	pixel_shift = angle_res/am_circ_deg
	for shift_nr in range(1,am_circ_deg):
		shift = int(shift_nr*pixel_shift)
		pixels[:,shift:] = pixels_n[:,:-shift]
		pixels[:,:shift] = pixels_n[:,-shift:]
		print("Constructing rectified image " + str(idx+shift_nr*am_imgs))
		cv2.imwrite("rectified_images/rectified-img-"+str(idx+shift_nr*am_imgs)+".jpg",pixels)