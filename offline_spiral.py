#!/usr/bin/env python3

'''
Easy all-in-one script to run Flightmare and BearingNet (offline) 
Author: mfirlefyn
Date: 23th of February 2024
Correspondence: mfirlefyn Github Issues
Paper reference: Direct Learning of Home Vector Direction for Insect-inspired Robot Navigation
'''

# MAKE SURE FILES ARE SOURCED
	# terminal 1: roscore
# RUN "earlyoom" TO PROTECT AGAINST RAM OVERUSE (not best solution, but helps to keep mess manageable)
	# terminal 2: earlyoom
# RUN BEARINGNET
	# terminal 3: source ros setup.bash files 
	# terminal 3: before invoking script, run: "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python3"
	# terminal 3: offline.py script

# import modules
import roslaunch
import rospy
import subprocess
import shutil
import os

# SCRIPT IS DIVIDED INTO BLOCKS WHICH CAN BE EASILY COMMENT TOGGLED OUT TO SELECT WHICH BLOCKS YOU WANT TO RUN DEPENDING ON WHICH PARTS YOU REQUIRE
print("=====ITERATING OVER QUAD DISTANCES IN GRID=====")
print("+++++PROGRAM START+++++")

# setting
x = 0		# row nr of nest (rel to grid, not distance)
y = 0		# col nr of nest (rel to grid, not distance)
am_gaze_angles = 360

# can be used to run the script for multiple distances
for dist in [1]: 	
	# change to the directory and generate the labels.csv file
	print(f"({x},{y},{dist})+++++GENERATE LABELS.CSV+++++")
	os.chdir("./bearingNet")
	subprocess.call(["python3", "writeLabels.py", "--nest_location_x", str(x), "--nest_location_y", str(y), "--grid_distance", str(dist), "--training_trajectory", "spiral"])
	os.chdir("./..")

	# run flightmare
	print(f"({x},{y},{dist})+++++RUN FLIGHTMARE+++++")
	cli_args = ["~/catkin_ws/src/flightmare/flightros/launch/camera/camera.launch",'dist:='+str(dist)]
	roslaunch_args = cli_args[1:]
	roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0],roslaunch_args)]
	rospy.init_node("FLIGHTMARE", anonymous=True)
	uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
	roslaunch.configure_logging(uuid)
	launch = roslaunch.parent.ROSLaunchParent(uuid,roslaunch_file)
	launch.start()
	rospy.sleep(300)
	launch.shutdown()

	# move images from Flightmare generation folder to rectification folder
	print(f"({x},{y},{dist})+++++MOVE OMNI-DIRECTIONAL IMAGES TO RECTIFICATION+++++")
	for image in os.listdir("~/catkin_ws/src/flightmare/flightros/src/camera/images"):
		shutil.copy2(os.path.join("~/catkin_ws/src/flightmare/flightros/src/camera/images",image),os.path.join("./rectification/images",image))

	# rectify the omni-directional images
	print(f"({x},{y},{dist})+++++RECTIFICATION OF OMNI-DIRECTIONAL IMAGES+++++")
	os.chdir("./rectification")
	subprocess.call(["python3", "rectify.py"])
	os.chdir("./..")

	# change image names with single digit to double digit
	print(f"({x},{y},{dist})+++++LET ALL RECTIFIED IMAGE NAMES CONTAIN DOUBLE DIGITS+++++")
	for i in range(10):
		shutil.move("./rectification/rectified_images/rectified-img-"+str(i)+".jpg","./rectification/rectified_images/rectified-img-0"+str(i)+".jpg")

	# move images from rectification folder to training folder
	print(f"({x},{y},{dist})+++++MOVE OMNI-DIRECTIONAL IMAGES TO DATALOADER+++++")
	for image in os.listdir("./rectification/rectified_images"):
		shutil.copy2(os.path.join("./rectification/rectified_images",image),os.path.join("./bearingNet/img_dir",image))

	# remove nest location from the image directory
	print(f"({x},{y},{dist})+++++MOVE NEST IMAGES OUT OF CNN IMAGE FOLDER+++++")
	nest_image1 = "./bearingNet/img_dir/rectified-img-" + str(x) + str(y) + ".jpg"
	shutil.move(nest_image1, "./bearingNet/rectified-img-" + str(x) + str(y) + ".jpg")
	for angle_idx in range(1,am_gaze_angles):
		nest_image = "./bearingNet/img_dir/rectified-img-" + str(angle_idx) + str(x) + str(y) + ".jpg"
		shutil.move(nest_image, "./bearingNet/rectified-img-" + str(angle_idx) + str(x) + str(y) + ".jpg")

	# run the bearingNet.py script in order to train the network
	print(f"({x},{y},{dist})+++++TRAIN THE NETWORK+++++")
	os.chdir("./bearingNet")
	subprocess.call(["python3", "bearingNet_offline_spiral.py", "--activity", "train", "--nest_location_x", str(x), "--nest_location_y", str(y)])
	os.chdir("./..")

	# run the logging script to log the train results (run and weight)
	print(f"({x},{y},{dist})+++++LOG THE NET WEIGHTS AND TENSORBOARD RUN+++++")
	os.chdir("./logging")
	subprocess.call(["python3", "logRun_offline_spiral.py", "--nest_location_x", str(x), "--nest_location_y", str(y), "--grid_distance", str(dist)])
	os.chdir("./..")

	# run the bearingNet.py script in order to evaluate the network
	print(f"({x},{y},{dist})+++++EVALUATE THE NETWORK+++++")
	os.chdir("./bearingNet")
	subprocess.call(["python3", "bearingNet_offline_spiral.py", "--activity", "evaluate", "--nest_location_x", str(x), "--nest_location_y", str(y)])
	os.chdir("./..")

	# run the logging script to log the evaluation results (graphs)
	print(f"({x},{y},{dist})+++++LOG THE EVALUATION GRAPHS+++++")
	os.chdir("./logging")
	subprocess.call(["python3", "logRun_offline_spiral.py", "--nest_location_x", str(x), "--nest_location_y", str(y), "--grid_distance", str(dist)])
	os.chdir("./..")

	# move the nest location back into image directory
	print(f"({x},{y},{dist})+++++MOVE NEST IMAGES BACK INTO CNN IMAGE FOLDER+++++")
	shutil.move("./bearingNet/rectified-img-" + str(x) + str(y) + ".jpg", nest_image1)
	for angle_idx in range(1,am_gaze_angles):
		nest_image = "./bearingNet/img_dir/rectified-img-" + str(angle_idx) + str(x) + str(y) + ".jpg"
		shutil.move("./bearingNet/rectified-img-" + str(angle_idx) + str(x) + str(y) + ".jpg", nest_image)

print("+++++PROGRAM COMPLETE+++++")