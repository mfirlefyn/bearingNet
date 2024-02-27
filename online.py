#!/usr/bin/env python3

'''
Easy all-in-one script to run BearingNet (online), Flightmare needs to be run separately
Author: mfirlefyn
Date: 23th of February 2024
Correspondence: mfirlefyn Github Issues
Paper reference: Direct Learning of Home Vector Direction for Insect-inspired Robot Navigation
'''

# MAKE SURE FILES ARE SOURCED
    # terminal 1: source ros setup.bash files and run flightmare
# RUN "earlyoom" TO PROTECT AGAINST RAM OVERUSE (not best solution, but helps to keep mess manageable)
    # terminal 2: earlyoom
# RUN BEARINGNET
    # terminal 3: offline.py script
    # before invoking script, run: "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"

# import modules
import subprocess
import shutil
import zmq
import os

print("=====ONLINE BEARINGNET=====")
print("+++++PROGRAM START+++++")

print("+++++WAIT FOR START TRAIN ZMQ MESSAGE+++++")
# get zmq messages
context = zmq.Context() # ZeroMQ Context

# Define the socket using the "Context"
sock = context.socket(zmq.SUB)
sock.setsockopt_string(zmq.SUBSCRIBE,"Train")
sock.connect("tcp://127.0.0.1:10255")

# Run a simple "Echo" server, break whenever Flightmare sends message that all training images are generated
try:
    while True:
        message = sock.recv()
        print("Echo: " + str(message))
        if str(message) == "b'start'":
            print("go")
            break

except KeyboardInterrupt:
    sock.close()


# setting
x = 5      # row nr of nest (rel to grid, not distance)
y = 5      # col nr of nest (rel to grid, not distance)

# can be used to run the script for multiple distances
for dist in [1]:   
    # change to the directory and generate the labels.csv file
    os.chdir("./bearingNet")
    subprocess.call(["python", "writeLabels.py", "--nest_location_x", str(x), "--nest_location_y", str(y), "--grid_distance", str(dist), "--training_trajectory", "spiral"])
    os.chdir("./..")

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
    subprocess.call(["python", "bearingNet_online.py", "--activity", "train", "--nest_location_x", str(x), "--nest_location_y", str(y)])
    os.chdir("./..")

    # run the logging script to log the train results (run and weight)
    print(f"({x},{y},{dist})+++++LOG THE NET WEIGHTS AND TENSORBOARD RUN+++++")
    os.chdir("./logging")
    subprocess.call(["python", "logRun_online.py", "--nest_location_x", str(x), "--nest_location_y", str(y), "--grid_distance", str(dist)])
    os.chdir("./..")

    # move the nest location back into image directory
    print(f"({x},{y},{dist})+++++MOVE NEST IMAGES BACK INTO CNN IMAGE FOLDER+++++")
    shutil.move("./bearingNet/rectified-img-" + str(x) + str(y) + ".jpg", nest_image1)
    for angle_idx in range(1,am_gaze_angles):
        nest_image = "./bearingNet/img_dir/rectified-img-" + str(angle_idx) + str(x) + str(y) + ".jpg"
        shutil.move("./bearingNet/rectified-img-" + str(angle_idx) + str(x) + str(y) + ".jpg", nest_image)

print("+++++WAIT FOR EVALUATION ZMQ MESSAGE+++++")
# get zmq messages
context = zmq.Context() # ZeroMQ Context

# Define the socket using the "Context"
sock = context.socket(zmq.SUB)
sock.setsockopt_string(zmq.SUBSCRIBE,"Evaluate")
sock.connect("tcp://127.0.0.1:10255")

counter = 0
# When evaluation image has come in, Python receives message and can rectify and evaluate it
try:
    while True:
        message = sock.recv()
        print("Echo: " + str(message))
        if str(message) == "b'start'":
            print("Evaluating current image")

            # rectify the omni-directional images
            print(f"({x},{y},{dist})+++++RECTIFICATION OF OMNI-DIRECTIONAL EVALUATION IMAGE+++++")
            os.chdir("./rectification")
            subprocess.call(["python", "rectify_single.py", "--image", "~/catkin_ws/src/flightmare/flightros/src/camera/eval_images/omni-dir-eval-img.jpg"])
            os.chdir("./..")

            # move images from rectification folder to evaluation folder
            print(f"({x},{y},{dist})+++++MOVE RECTIFIED IMAGE TO EVALUATION FOLDER+++++")
            shutil.move("./rectification/rectified-eval-img.jpg","~/catkin_ws/src/flightmare/flightros/src/camera/eval_images/rectified-eval-img.jpg")

            # run the bearingNet.py script in order to evaluate the network
            print(f"({x},{y},{dist})+++++EVALUATE THE NETWORK ON ONLINE IMAGE+++++")
            os.chdir("./bearingNet")
            subprocess.call(["python", "bearingNet_online.py", "--activity", "evaluate", "--nest_location_x", str(x), "--nest_location_y", str(y)])

            print("============================================")

            counter+=1

    sock.close()

except KeyboardInterrupt:
    sock.close()

print("+++++PROGRAM COMPLETE+++++")