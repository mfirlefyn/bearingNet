'''
Log all evaluation plots from offline rectGrid bearingNet script 
Author: mfirlefyn
Date: 23th of February 2024
Correspondence: mfirlefyn Github Issues
Paper reference: Direct Learning of Home Vector Direction for Insect-inspired Robot Navigation
'''

# import modules
import os
import shutil
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-x", "--nest_location_x", required = True, help = "x coordinate of the nest")
ap.add_argument("-y", "--nest_location_y", required = True, help = "y coordinate of the nest")
ap.add_argument("-d", "--grid_distance", required = True, help = "incremental distance in between quadcopter objects")
args = vars(ap.parse_args())

# define some functions to get most recent network weights, which are logged in a folder named according to tensorboard summary writer convention
def convertMonthToNumber(month):
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    num_map = dict([(y,x+1) for x,y in enumerate(months)])

    return(num_map[month])

def convertNameToNumber(name):
    first = name.split("_")
    second = first[1].split("-")

    month_number = convertMonthToNumber(first[0][:3]) * 30 * 24* 60 * 60
    day_number = int(first[0][3:]) * 24 * 60 * 60
    hour_number = int(second[0]) * 60 * 60
    min_number = int(second[1]) * 60
    sec_number = int(second[2]) 

    return(month_number + day_number + hour_number + min_number + sec_number)

def getNameLatestRun(dir):
    names = os.listdir(dir)
    runs_number = [convertNameToNumber(name) for name in names]

    return(names[runs_number.index(max(runs_number))])

def getDirNameFromLatestRun(latest):
    name = latest.split("_")

    return(name[0] + "_" + name[1])

# define function that writes all of the settings to a file, this is meant as a template, be sure to adjust it accordingly
def writeSettings(path, loc_x, loc_y, distance):
    with open(os.path.join(path,"tunedParams.md"), 'w') as f:
        f.write(f"* nest location: {loc_x},{loc_y}\n")
        f.write("* grid distance: "+str(10*int(distance))+"x"+str(10*int(distance))+" m\n")
        f.write("* grid amount: 100\n")
        f.write("* image dims: 1800x201\n")
        f.write("\n")
        f.write("* CNN arch:\n")
        f.write("\t * conv layer 1: (1,2,5,stride=4), tanh activation\n")
        f.write("\t * conv layer 2: (2,4,5,stride=4), tanh activation\n")
        f.write("\t * flatten: (x,1)\n")
        f.write("\t * fully conn layer 1: (5376,2), tanh activation\n")
        f.write("* lr = 0.9e-3, scheduler: ReduceLROnPlateau(optimizer, mode='min', factor=0.98, patience=10, min_lr=1e-12, eps=1e-14, verbose=True)\n")
        f.write("* nr. training samples: 396, random sampled\n")
        f.write("* epochs: 100\n")
        f.write("\n")
        f.write("note: online spiral, R=0+0.25*theta, 25 samples on 4 turns, step size = 0.25, 1.5*gaussian")

# define a function to copy the log files from generated directory to log 
def logRunFiles(oldPath,dest,logDirPath):
    newPath = os.path.join(logDirPath,dest)
    shutil.move(oldPath,newPath)

# get latest directory that has been made by tensorflow
runs_dir = os.path.expanduser("./logs")

# Toggle comment dependent on whether you want to automatically log latest run or manually log an earlier run
latest_run = getNameLatestRun(runs_dir) # automatic, latest
#latest_run = "Aug24_11-47-44"          # manual

latest_run_path = os.path.join(runs_dir,latest_run)

# settings
amount_of_gaze_angles = 360

# path settings
weights_path = os.path.expanduser("../bearingNet/bearing_net.pth")  # get the path the the pytorch weight file of most current run
loss_path = os.path.expanduser("../bearingNet/min_running_loss.md") # get the path the the pytorch minimum loss value file of most current run
labels_in_epoch_path = os.path.expanduser("../bearingNet/labels_in_epoch.md")   # get the path the the pytorch labels for minimum loss value file of most current run

# get the path to the evaluation graphs
error_run_path = os.path.expanduser("../bearingNet/error_run.png")
error_run_norm_label_path = os.path.expanduser("../bearingNet/error_run_norm_label.png")
error_run_norm_dist_path = os.path.expanduser("../bearingNet/error_run_norm_dist.png")
#positions_in_epoch_path = os.path.expanduser("../bearingNet/positions_in_epoch.png")
#gaze_angles_in_epoch_path = os.path.expanduser("../bearingNet/gaze_angles_in_epoch.png")
MSE_error_wrt_label_path = os.path.expanduser("../bearingNet/MSE_error_wrt_label.png")
MSE_error_wrt_pos_path = os.path.expanduser("../bearingNet/MSE_error_wrt_pos.png")
threeD_error_wrt_label_pos_path = os.path.expanduser("../bearingNet/threeD_error_wrt_label_pos.png")

# get the path the the pytorch minimum evaluation loss value file of most current run
eval_loss_path = os.path.expanduser("../bearingNet/min_eval_loss.md")

# get log directory in docs runs
logs_dir = os.path.expanduser("./logs")
log_dir = getDirNameFromLatestRun(latest_run)
log_dir_path = os.path.join(logs_dir,log_dir)

# make a new log directory with the same name as the latest run, copy run and weights, after train cycle
if not os.path.exists(log_dir_path):
    os.mkdir(log_dir_path)

    # copy the latest run (tensorboard) to the log directory
    logRunFiles(latest_run_path,latest_run,log_dir_path)

    # copy the minimum running loss to the log directory
    logRunFiles(loss_path,"min_running_loss.md",log_dir_path)

    # copy the labels for minimum running loss to the log directory
    logRunFiles(labels_in_epoch_path,"labels_in_epoch.md",log_dir_path)

    # write the tuned params settings to file
    writeSettings(log_dir_path,args["nest_location_x"],args["nest_location_y"],args["grid_distance"])

# evaluation cycle, directory already exist because it is needed to evaluate
else:
    # copy the weights to the log directory
    logRunFiles(weights_path,"bearing_net.pth",log_dir_path)

    logRunFiles(error_run_path,"error_run.png",log_dir_path)

    logRunFiles(error_run_norm_label_path,"error_run_norm_label.png",log_dir_path)

    logRunFiles(error_run_norm_dist_path,"error_run_norm_dist.png",log_dir_path)
        
    # 2d plots need to be logged for every gaze angle
    for i in range(amount_of_gaze_angles):
        # error run 2d
        logRunFiles(os.path.expanduser("../bearingNet/error_run_2d_" + str(i) + ".png"),"error_run_2d_" + str(i) + ".png",log_dir_path)

        # error run norm label 2d
        logRunFiles(os.path.expanduser("../bearingNet/error_run_norm_label_2d_" + str(i) + ".png"),"error_run_norm_label_2d_" + str(i) + ".png",log_dir_path)

        # error run norm dist 2d
        logRunFiles(os.path.expanduser("../bearingNet/error_run_norm_dist_2d_" + str(i) + ".png"),"error_run_norm_dist_2d_" + str(i) + ".png",log_dir_path)

        # grid error (MSE or MAE)
        logRunFiles(os.path.expanduser("../bearingNet/grid_error_" + str(i) + ".png"),"grid_error_" + str(i) + ".png",log_dir_path)


    logRunFiles(MSE_error_wrt_label_path,"MSE_error_wrt_label.png",log_dir_path)

    logRunFiles(MSE_error_wrt_pos_path,"MSE_error_wrt_pos.png",log_dir_path)

    logRunFiles(threeD_error_wrt_label_pos_path,"threeD_error_wrt_label_pos.png",log_dir_path)

    # copy the minimum eval loss to the log directory
    logRunFiles(eval_loss_path,"min_eval_loss.md",log_dir_path)