'''
Write the training labels dependent on rectangular grid or spiral training trajectory
Author: mfirlefyn
Date: 23th of February 2024
Correspondence: mfirlefyn Github Issues
Paper reference: Direct Learning of Home Vector Direction for Insect-inspired Robot Navigation
'''

# import modules
import argparse
import os

# settings
src_dir = os.path.expanduser("./img_dir")   # path to the source images
images = os.listdir(src_dir)                # getting all the files from the source directory
am_gaze_angles = 360
am_positions = 100

# filter out anything that is not relevant
images[:] = [name for name in images if any(word in name for word in ["rectified-img"])]
# sort the list of images
images.sort(key = lambda x: int(x.split(".")[0].split("-")[-1]))

# write annotation file (angles are [-1,1](*pi) rad)
import numpy as np
import csv

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-x", "--nest_location_x", required = True, help = "x coordinate of the nest")
ap.add_argument("-y", "--nest_location_y", required = True, help = "y coordinate of the nest")
ap.add_argument("-d", "--grid_distance", required = True, help = "incremental distance in between quadcopter objects")
ap.add_argument("-t", "--training_trajectory", required = True, help = "either 'rectGrid' or 'spiral'")
args = vars(ap.parse_args())

# initialize the grid, left top is (0,0), all coords go to the right and down (like the quads are spawned)
gridSize = 10
gridDist = int(args["grid_distance"])
grid = []

# training trajectory choice
traj = args["training_trajectory"]

# construct points for rectGrid
if traj == "rectGrid":
    for y in range(gridSize):
        for x in range(gridSize):
            grid.append([gridDist*x,gridDist*y])
elif traj == "spiral":
    pointsPerRotation = 25
    numRotations = 4
    spiralParam1 = 0
    spiralParam2 = 0.25

    for theta in np.linspace(0,numRotations*2*np.pi,numRotations*pointsPerRotation,endpoint=False):
        radius = spiralParam1 + spiralParam2*theta
        x = radius*np.cos(theta)
        y = radius*np.sin(theta)
        grid.append([-y,x])
else:
    print("Given 'training_trajectory' argument is incorrect. Valid choices: 'rectGrid' and 'spiral'")
    print("Exiting program.")
    exit()

grid = np.array(grid)

# Calculate the angle to the nest by determining relative angle between absolute angle and gaze orientation
def angleToNest(nest,orientation):
    anglesToNest = []
    relVectorsToNest = np.add(-grid,nest)                               # x,y
    relVectors = np.array([np.cos(orientation),np.sin(orientation)])    # x,y

    # rotate reference frame according to one angle to determine the other
    for i in range(gridSize**2):
        dot = relVectors[0]*relVectorsToNest[i][0] + relVectors[1]*relVectorsToNest[i][1]
        det = relVectors[0]*relVectorsToNest[i][1] - relVectors[1]*relVectorsToNest[i][0]
        anglesToNest.append(np.arctan2(det,dot))
    
    return(np.resize(np.array(anglesToNest),(gridSize,gridSize)))

# set nest location
nest_loc = np.array([gridDist*int(args["nest_location_x"]),gridDist*int(args["nest_location_y"])])
nest_loc_idx = (nest_loc[0]/gridDist*10+nest_loc[1]/gridDist)
nest_loc = np.flip(nest_loc)    # different mapping for nest location (from (rows,cols) to (x,y))
# normal distribution for label noise (only available when toggled later on)
mu, sigma = 0, 10*np.pi/180     # mean and std dev of 10 deg

# open the file in the write mode
with open(os.path.expanduser("labels.csv"), 'w') as f:
    # create the csv writer
    writer = csv.writer(f)

    # write the header
    writer.writerow(["name","labelx","labely"])

    for angle_idx in range(am_gaze_angles):
        angle = angle_idx*np.pi/180+np.pi/2
        tolerance = 1e-16
        relAngles = np.ndarray.flatten(angleToNest(nest_loc,angle))
        relAngles[abs(relAngles)<tolerance] = 0.
        relAngles[relAngles==np.pi] = -np.pi        # 180deg gaze is ambiguous, only negative is valid (choice)

        for i in range(int(len(images)/am_gaze_angles)):
            # write a row to the csv file
            if i != nest_loc_idx:
                # Toggle which lines are written depending on labels with or without noise
                writer.writerow([images[i+angle_idx*am_positions],np.cos(relAngles[i]),np.sin(relAngles[i])])   # without noise
                #writer.writerow([images[i+angle_idx*am_positions],np.cos(relAngles[i]+np.random.normal(mu,sigma)),np.sin(relAngles[i]+np.random.normal(mu,sigma))])    # with noise