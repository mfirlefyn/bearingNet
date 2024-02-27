'''
Train and evaluate readily available rectified catadioptric images (offline) 
Author: mfirlefyn
Date: 23th of February 2024
Correspondence: mfirlefyn Github Issues
Paper reference: Direct Learning of Home Vector Direction for Insect-inspired Robot Navigation
Code reference url 1: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
Code reference url 2: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
'''

# import dataset object modules
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

# making a separate object to hold the custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        img_name = self.img_labels.iloc[idx, 0]
        labelx = self.img_labels.iloc[idx, 1]
        labely = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labelx = self.target_transform(labelx)
            labely = self.target_transform(labely)

        return image, labelx, labely, img_name

# import some more modules
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# dataset settings
batchSize = 1

# training dataset object definition
training_data = CustomImageDataset("labels.csv","img_dir")

# import NN modules
import torch.nn as nn
from torch import tanh

# define the CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 5, stride=4)
        self.conv2 = nn.Conv2d(2, 4, 5, stride=4)
        self.fc1 = nn.Linear(5376, 2)   # (4752,2) for 201x1600, (5376,2) for 201x1800
        self.fc2 = nn.Linear(1, 1)

    def forward(self, x):
        x = tanh(self.conv1(x))
        x = tanh(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = tanh(self.fc1(x))
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            weights = module.weight.data
            nn.init.xavier_normal_(weights,gain=nn.init.calculate_gain('tanh'))
            if module.bias is not None:
                module.bias.data.zero_()

net = Net()

# Settings and definition to train and evaluate the network
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import argparse

# define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.9e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.98, patience=10, min_lr=1e-12, eps=1e-14, verbose=True)

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

# define some functions to be used in plotting the evaluation results, x and y are the home vector coordinates, lists contain coordinated for multiple locations (per image)
def Calc_coords_wrt_north(lst):
    x_coords = -1 * R * np.sin(np.array(lst))        # north
    y_coords = R * np.cos(np.array(lst))             # north

    return x_coords,y_coords

def Calc_coords_wrt_north_from_gaze_angle(xlst,ylst,gaze_angle_sublst):
    xlst = np.array(xlst)
    ylst = np.array(ylst)
    gaze_angle_sublst = np.array(gaze_angle_sublst)+np.pi/2
    # reference frame rotation
    x_coords = R*np.cos(gaze_angle_sublst)*xlst - R*np.sin(gaze_angle_sublst)*ylst     
    y_coords = R*np.sin(gaze_angle_sublst)*xlst + R*np.cos(gaze_angle_sublst)*ylst          

    return x_coords,y_coords

def Calc_angle_deviation_in_deg(xlst,ylst,angle_sublst):
    xlst = np.array(xlst)
    ylst = np.array(ylst)
    # reference frame rotation
    x_coords = np.cos(angle_sublst)*xlst + np.sin(angle_sublst)*ylst
    y_coords = -np.sin(angle_sublst)*xlst + np.cos(angle_sublst)*ylst          

    return np.abs(np.arctan2(y_coords,x_coords))*180/np.pi

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--activity", required = True, help = "activity string indicating evaluation or train")
ap.add_argument("-x", "--nest_location_x", required = True, help = "x coordinate of the nest")
ap.add_argument("-y", "--nest_location_y", required = True, help = "y coordinate of the nest")
args = vars(ap.parse_args())

# get latest directory that has been made by tensorflow
runs_dir = os.path.expanduser("logging/runs")
# get log directory in docs runs
logs_dir = os.path.expanduser("logging/logs")

latest_run = getNameLatestRun(runs_dir)
log_dir = getDirNameFromLatestRun(latest_run)
log_dir_path = os.path.join(logs_dir,log_dir)

# Toggle comment dependent on whether you want to manually select weights or latest trained weights
evalPATH = os.path.join(log_dir_path,"bearing_net.pth")     # automatic, latest
#evalPATH = "logging/logs/Aug24_11-47-44/bearing_net.pth"   # manual 


# train or evaluate
activity = args["activity"]

# set extra high in order to make sure that the model is saved during the first epoch
prev_loss = 100000000 

if activity == "train":
    # define tensorboard summary writer to keep track of loss over multiple epochs
    writer = SummaryWriter()

    # lists to calculate MAE during training
    train_labelx_lst = []
    train_labely_lst = []
    train_outputx_lst = []
    train_outputy_lst = []

    # train the network
    for epoch in range(1):  # 1 epoch for oneshot
        running_loss = 0.0

        # randomly shuffle the dataset, can be reshuffled per epoch, otherwise put before for loop
        train_dataloader = DataLoader(training_data, batch_size=batchSize, shuffle=True)

        # hold results temporarily to be stored for best result 
        labels_in_epoch = []
        running_train_outputx_lst = []
        running_train_outputy_lst = []
        train_labelx_lst = []
        train_labelx_lst = []

        # start training
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs, labels, and names
            inputs, labelx, labely, names = data
            labels = torch.cat((labelx,labely),0).unsqueeze(1).transpose(1,0)
            # inputs are 0,255. Pytorch expects 0,1
            inputs = inputs.float()/255.
            # keeping track of the labels
            labels_in_epoch.append(names)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward propagation
            outputs = net(inputs)
            # store output and labels
            running_train_outputx_lst.append(outputs[0,0].item())
            running_train_outputy_lst.append(outputs[0,1].item())
            train_labelx_lst.append(labels[0,0].item())
            train_labely_lst.append(labels[0,1].item())
            # backward propagation + optimize
            loss = criterion(outputs, labels.float())       # convert x,y home vector labels to float
            loss.backward()
            optimizer.step()

            # accumulate running loss over data
            running_loss += loss.item()

        # print accumulated loss over epoch
        print(f'[{epoch + 1}, {len(labels_in_epoch) + 1:5d}] loss: {running_loss}')

        # tensorboard graph
        writer.add_scalar('Loss/train', running_loss, epoch)

        # save best output, labels, and weights of net
        if running_loss < prev_loss:
            prev_loss = running_loss
            train_outputx_lst = running_train_outputx_lst
            train_outputy_lst = running_train_outputy_lst
            train_label_angle_lst = np.arctan2(np.array(train_labely_lst),np.array(train_labelx_lst))
            # save the trained model
            PATH = './bearing_net.pth'
            torch.save(net.state_dict(), PATH)
            print('Model saved!')

            with open("labels_in_epoch.md", 'w') as f:
                f.write(f"{labels_in_epoch}")       # strings

        # before next epoch starts, run scheduler to update learning rate and reset accumulated loss
        scheduler.step(running_loss)
        running_loss = 0.0

    # save best accumulated loss (MSE) [m2] and calculated MAE [deg]
    with open("min_running_loss.md", 'w') as f:
        f.write(f"* smallest running_loss during training = {prev_loss}\n")
        f.write(f"* smallest running_loss during training in deg = {np.sum(Calc_angle_deviation_in_deg(np.array(train_outputx_lst),np.array(train_outputy_lst),train_label_angle_lst),axis=0)}")

    print('Finished Training')

elif activity == "evaluate":
    # loading best model 
    net.load_state_dict(torch.load(evalPATH))

    # set nest location from arguments and convert to data list index
    nest_loc = np.array([int(args["nest_location_x"]),int(args["nest_location_y"])])
    nest_loc_idx = (nest_loc[0]*10+nest_loc[1])
    # different mapping for nest location (from (rows,cols) to (x,y))
    nest_loc = np.flip(nest_loc)

    # settings
    gridSize = 10                   # grid size for the grid of quads
    stepSize = 1                    # step size for the grid of quads
    amount_of_gaze_angles = 360     # amount of gaze angles that need to be evaluated, included in plots

    labelx_lst = []
    labely_lst = []
    outputx_lst = []
    outputy_lst = []
    sample_index_lst = []
    gaze_angle_lst = []
    id_lst = []

    print("Evaluating datapoints")

    # does not need to be reshuffled during evaluation
    train_dataloader = DataLoader(training_data, batch_size=batchSize, shuffle=False)

    # evaluate the network
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs, labels, and names
        inputs, labelx, labely, names = data
        labels = torch.cat((labelx,labely),0).unsqueeze(1).transpose(1,0)
        # inputs are 0,255. Pytorch expects 0,1
        inputs = inputs.float()/255.
        # forward propagation
        outputs = net(inputs)
        # store output and labels
        outputx_lst.append(outputs[0,0].item())
        outputy_lst.append(outputs[0,1].item())
        labelx_lst.append(labels[0,0].item())
        labely_lst.append(labels[0,1].item())
        label_angle_lst = np.arctan2(np.array(labely_lst),np.array(labelx_lst))
        sample_index_lst.append(i)

        # extract gaze angles from image names
        image_id = names[0].split("-")[-1].split(".")[0]
        id_lst.append(image_id)
        if len(image_id) == 2:
            gaze_angle_lst.append(0)
        else:
            gaze_angle_lst.append(int(image_id[:-2])*np.pi/180)

    # save best calculated MAE [deg]
    with open("min_eval_loss.md", 'w') as f:
        f.write(f"* loss during evaluation = {np.sum(Calc_angle_deviation_in_deg(np.array(outputx_lst),np.array(outputy_lst),label_angle_lst),axis=0)}")

    # the error angle should be the angle between the label and the prediction
    dot = np.array(labelx_lst)*np.array(outputx_lst) + np.array(labely_lst)*np.array(outputy_lst)
    det = np.array(labelx_lst)*np.array(outputy_lst) - np.array(labely_lst)*np.array(outputx_lst)
    evaluation_error = np.arctan2(det,dot)
    evaluation_error_deg = evaluation_error*180/np.pi

    # Plot and save degree error plot (not normalized)
    print("Save error_run")
    
    plt.figure()
    plt.title('Error between output (y*) and label (y)')
    plt.xlabel('image ID [-]')
    plt.ylabel('error: (y-y*)*180 [deg]')
    plt.plot(sample_index_lst,evaluation_error_deg)

    plt.savefig("error_run.png")
    
    # Plot and save 2d error map over grid for every gaze angle
    print("Save error_run_2d")

    for i in range(amount_of_gaze_angles):
        evaluation_error_deg_2d = np.insert(evaluation_error_deg[i*99:(i+1)*99]**2,nest_loc_idx,0.).reshape(gridSize,gridSize)
        # Plot settings
        fig = plt.figure(figsize=(5,5),dpi=300)
        plt.ylabel('offset from origin [idx]')
        ax = plt.gca()
        ax.set_aspect('equal','box')
        im = ax.pcolormesh(np.arange(gridSize+1), np.arange(gridSize+1), evaluation_error_deg_2d, shading='flat', vmin=evaluation_error_deg_2d.min(), vmax=evaluation_error_deg_2d.max(), cmap=cm.gray)  
        ax.add_patch(Rectangle((nest_loc[0],nest_loc[1]), 1, 1, fc='none', ec='red', lw=2))
        axins = inset_axes(ax,width="100%",height="5%",loc='upper center',borderpad=-2)
        cbar = plt.colorbar(im,cax=axins,fraction=0.046,pad=0.04,ticks=[evaluation_error_deg_2d.min(),evaluation_error_deg_2d.max()],orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        cbar.set_label('MSE error [deg^2]',x=0.5,labelpad=-35)

        plt.savefig("error_run_2d_" + str(i) + ".png",bbox_inches='tight',pad_inches=0.3)

        plt.close()

    
    # Plot and save error plot, normalized by label
    print("Save error_run_norm_label")

    labelx_div_lst = np.array(labelx_lst)
    labely_div_lst = np.array(labely_lst)
    label_div_lst = np.arctan2(labely_div_lst,labelx_div_lst)
    # replace 0 with low number in order to prevent division by 0
    label_div_lst[label_div_lst==0] = 1e6       # normalizing with label will not work since there will be division by zero
    evaluation_error_norm_angle = evaluation_error/label_div_lst   
    # Plot settings
    plt.figure()
    plt.title('Error between output (y*) and label (y)')
    plt.xlabel('image ID [-]')
    plt.ylabel('error: (y-y*)/y')
    plt.plot(sample_index_lst,evaluation_error_norm_angle)

    plt.savefig("error_run_norm_label.png")
    
    # Plot and save 2d error map over grid, normalized by label, for every gaze angle
    print("Save error_run_norm_label_2d")

    for i in range(amount_of_gaze_angles):
        evaluation_error_norm_angle_2d = np.insert(evaluation_error_norm_angle[i*99:(i+1)*99]**2,nest_loc_idx,0.).reshape(gridSize,gridSize)
        # Plot settings
        fig = plt.figure(figsize=(5,5),dpi=300)
        plt.ylabel('offset from origin [idx]')
        ax = plt.gca()
        ax.set_aspect('equal','box')
        im = ax.pcolormesh(np.arange(gridSize+1), np.arange(gridSize+1), evaluation_error_norm_angle_2d, shading='flat', vmin=evaluation_error_norm_angle_2d.min(), vmax=evaluation_error_norm_angle_2d.max(), cmap=cm.gray)
        ax.add_patch(Rectangle((nest_loc[0],nest_loc[1]), 1, 1, fc='none', ec='red', lw=2))
        axins = inset_axes(ax,width="100%",height="5%",loc='upper center',borderpad=-2)
        cbar = plt.colorbar(im,cax=axins,fraction=0.046,pad=0.04,ticks=[evaluation_error_norm_angle_2d.min(),evaluation_error_norm_angle_2d.max()],orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        cbar.set_label('MSE error [-]',x=0.5,labelpad=-35)

        plt.savefig("error_run_norm_label_2d_" + str(i) + ".png",bbox_inches='tight',pad_inches=0.3)

        plt.close()
    

    # Plot and save error plot, normalized by distance to the nest
    print("Save error_run_norm_dist")

    dist_lst = []   # make list of distances to nest
    for i in range(amount_of_gaze_angles):
        for y in range(gridSize):
            for x in range(gridSize):
                if x != nest_loc[0] or y != nest_loc[1]:
                    dist = np.sqrt(stepSize*(x-nest_loc[0])**2+stepSize*(y-nest_loc[1])**2)
                    dist_lst.append(dist)
    
    dist_lst = np.array(dist_lst)
    dist_lst[dist_lst==0] = 1e6
    evaluation_error_norm_dist = evaluation_error/dist_lst
    # Plot settings
    plt.figure()
    plt.title('Error between output (y*) and label (y)')
    plt.xlabel('image ID [-]')
    plt.ylabel('error: (y-y*)/(nest distance) [deg/m]')
    plt.plot(sample_index_lst,evaluation_error_norm_dist)

    plt.savefig("error_run_norm_dist.png")
    
    # Plot and save 2d error map over grid, normalized by distanace to nest, for every gaze angle
    print("Save error_run_norm_dist_2d")

    for i in range(amount_of_gaze_angles):
        evaluation_error_norm_dist_2d = np.insert(evaluation_error_norm_dist[i*99:(i+1)*99]**2,nest_loc_idx,0.).reshape(gridSize,gridSize)
        # Plot settings
        fig = plt.figure(figsize=(5,5),dpi=300)
        plt.ylabel('offset from origin [idx]')
        ax = plt.gca()
        ax.set_aspect('equal','box')
        im = ax.pcolormesh(np.arange(gridSize+1), np.arange(gridSize+1), evaluation_error_norm_dist_2d, shading='flat', vmin=evaluation_error_norm_dist_2d.min(), vmax=evaluation_error_norm_dist_2d.max(), cmap=cm.gray)
        ax.add_patch(Rectangle((nest_loc[0],nest_loc[1]), 1, 1, fc='none', ec='red', lw=2))
        axins = inset_axes(ax,width="100%",height="5%",loc='upper center',borderpad=-2)
        cbar = plt.colorbar(im,cax=axins,fraction=0.046,pad=0.04,ticks=[evaluation_error_norm_dist_2d.min(),evaluation_error_norm_dist_2d.max()],orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.set_label('MSE error [deg^2/m^2]',x=0.5,labelpad=-35)

        plt.savefig("error_run_norm_dist_2d_" + str(i) + ".png",bbox_inches='tight',pad_inches=0.3)

        plt.close()
    
    
    # Plot and save 2d vector, confidence, and error map over grid for every gaze angle
    print("Saving grid vector, confidence, and error maps")

    # settings
    R = 0.06    # radius of the vectors
    x_interv = [538-1.,548.]
    y_interv = [573.-1,583.+1]
    treePos_lst = pd.read_csv("treepositions.csv")
    xTreePos = treePos_lst.iloc[:, 0].to_numpy()*1000.
    yTreePos = treePos_lst.iloc[:, 2].to_numpy()*1000.
    TreePos = np.transpose(np.append(xTreePos,yTreePos).reshape(2,len(xTreePos)))
    # making sure that the respective x and y coords stay together, x coords are linearly increasing while y coords are random
    TreePos = TreePos[(x_interv[0] <= TreePos[:,0]) & (TreePos[:,0] <= x_interv[1]) & (y_interv[0] <= TreePos[:,1]) & (TreePos[:,1] <= y_interv[1])]
    TreePos[:,0] -= x_interv[0] 
    TreePos[:,1] -= y_interv[0]
    # extracted tree positions also contain the positions of certain plants, their locs must be imported manually
    TreePos[:5,0] = [539.69111,538.0117,536.0624,541.9103,539.9611]
    TreePos[:5,1] = [569.2007,575.0488,578.9474,578.9474,580.8967]
    TreeFoliageRadius = [(5.17+4.88)/2,(4.99+4.94)/2,(4.47+4.38)/2,(4.56+4.62)/2,(6+6.35)/2]

    for i in range(amount_of_gaze_angles):
        # New figure to hold subfigures
        fig = plt.figure(figsize=(5,15),dpi=600)

        # Vector plot settings
        x_coords, y_coords = Calc_coords_wrt_north_from_gaze_angle(outputx_lst[i*99:(i+1)*99],outputy_lst[i*99:(i+1)*99],gaze_angle_lst[i*99:(i+1)*99])
        fig1 = plt.subplot(3,1,1)
        plt.xlabel('offset from origin [-]',fontsize=14)
        plt.ylabel('offset from origin [-]',fontsize=14)
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        ax.set_title('Bearing Map',fontsize=18)
       
        origins = [[],[]]        
        for y in np.arange(-5,5,stepSize):
            for x in np.arange(-5,5,stepSize):
                if x != (nest_loc[0]-5) or y != (nest_loc[1]-5):
                    origins[0].append(x)
                    origins[1].append(y)

        # Vector plot
        plt.quiver(*origins,x_coords,y_coords,scale=1,color='navy')
        plt.xlim([-5.5,4.5])
        plt.ylim([-5.5,4.5])
        plt.scatter(nest_loc[0]-5,nest_loc[1]-5,color='red',s=5)
    

        # Confidence plot settings
        confidence = np.abs(np.sqrt(np.array(outputx_lst[99*i:99*(i+1)])**2+np.array(outputy_lst[99*i:99*(i+1)])**2)-1)
        confidence = np.insert(confidence,nest_loc_idx,0.)
        conf_matrix = confidence.reshape(gridSize,gridSize) # use the scaled coords as indices to insert into confidence values

        fig2 = plt.subplot(3,1,2)
        plt.xlabel('offset from origin [-]',fontsize=14)
        plt.ylabel('offset from origin [-]',fontsize=14)
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        ax.set_title('Confidence',fontsize=18)
        plt.xticks([1.5,3.5,5.5,7.5,9.5],np.array([-4,-2,0,2,4]).astype('str'))
        plt.yticks([1.5,3.5,5.5,7.5,9.5],np.array([-4,-2,0,2,4]).astype('str'))
        im = ax.pcolormesh(np.arange(gridSize+1), np.arange(gridSize+1), conf_matrix, shading='flat', vmin=conf_matrix.min(), vmax=conf_matrix.max(),cmap=cm.binary)
        axins = inset_axes(ax,width="5%",height="100%",loc='center right',borderpad=-2)
        cbar = plt.colorbar(im,cax=axins)
        cbar.set_label('|1-vector length| [-]', rotation=90, labelpad=10, fontsize=14)

        ax.add_patch(Rectangle((nest_loc[0],nest_loc[1]), 1, 1, fc='none', ec='red', lw=2))
        

        # MSE error plot settings 
        MSE = ((np.insert(np.array(labelx_lst[i*99:(i+1)*99]),nest_loc_idx,0.)-np.insert(np.array(outputx_lst[i*99:(i+1)*99]),nest_loc_idx,0.))**2+(np.insert(np.array(labely_lst[i*99:(i+1)*99]),nest_loc_idx,1.)-np.insert(np.array(outputy_lst[i*99:(i+1)*99]),nest_loc_idx,1.))**2)
        label_angle_lst = np.arctan2(np.insert(np.array(labely_lst[i*99:(i+1)*99]),nest_loc_idx,0.),np.insert(np.array(labelx_lst[i*99:(i+1)*99]),nest_loc_idx,0.))
        MSE_angle = Calc_angle_deviation_in_deg(np.insert(np.array(outputx_lst[i*99:(i+1)*99]),nest_loc_idx,0.),np.insert(np.array(outputy_lst[i*99:(i+1)*99]),nest_loc_idx,0.),label_angle_lst)
        MSE_angle_matrix = MSE_angle.reshape(gridSize,gridSize)

        fig3 = plt.subplot(3,1,3)
        plt.xlabel('offset from origin [-]',fontsize=14)
        plt.ylabel('offset from origin [-]',fontsize=14)
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        ax.set_title('Error',fontsize=18)
        plt.xticks([1.5,3.5,5.5,7.5,9.5],np.array([-4,-2,0,2,4]).astype('str'))
        plt.yticks([1.5,3.5,5.5,7.5,9.5],np.array([-4,-2,0,2,4]).astype('str'))
        im = ax.pcolormesh(np.arange(gridSize+1), np.arange(gridSize+1), MSE_angle_matrix, shading='flat', vmin=MSE_angle_matrix.min(), vmax=MSE_angle_matrix.max(),cmap=cm.binary)
        axins = inset_axes(ax,width="5%",height="100%",loc='center right',borderpad=-2)
        cbar = plt.colorbar(im,cax=axins)
        cbar.set_label('|angle deviation| [$^\circ$]', rotation=90, labelpad=10, fontsize=14)
        # Subplot shifts to align 
        pos2 = fig2.get_position()
        pos2.y0 -= 0.01
        pos2.y1 -= 0.01 
        fig2.set_position(pos2)
        pos3 = fig3.get_position()
        pos3.y0 -= 0.02
        pos3.y1 -= 0.02 
        fig3.set_position(pos3)
        
        ax.add_patch(Rectangle((nest_loc[0],nest_loc[1]), 1, 1, fc='none', ec='red', lw=2))


        plt.savefig("grid_error_" + str(i) + ".png",bbox_inches='tight',pad_inches=0.2)

        plt.close()
    
    
    # Additional graphs for own evaluation
    ###############################################################################################################################
    # Plot MSE error wrt label
    print("Save MSE wrt label plot")

    id_label_err_lst = []   # make a list that has [id,label_angle,angle_MSE_error]
    label_angle = np.arctan2(np.array(labely_lst),np.array(labelx_lst))*180/np.pi      # degree
    output_angle = np.arctan2(np.array(outputy_lst),np.array(outputx_lst))*180/np.pi      # degree
    angle_error = np.abs(evaluation_error_deg)      # deg, this error returns high values since sometimes arctan2 returns e.g. 170deg for an angle of -180deg => very high error (prob same issue as single label)
    for i in range(len(id_lst)):
        id_label_err_lst.append([id_lst[i],label_angle[i],angle_error[i]])

    id_label_err_lst = np.array(id_label_err_lst)
    id_label_err_lst = id_label_err_lst[id_label_err_lst[:,1].astype(float).argsort()]        # sort rows on the value of the label

    # Plot settings
    fig = plt.figure(figsize=(10,10),dpi=300)
    plt.plot(id_label_err_lst[:,1].astype(float),id_label_err_lst[:,2].astype(float))
    plt.xlabel('label angle [deg]')
    plt.ylabel('error: abs(y-y*) [deg]')
    plt.xlim([id_label_err_lst[:,1].astype(float).min()-5,id_label_err_lst[:,1].astype(float).max()+5])
    plt.ylim([id_label_err_lst[:,2].astype(float).min()-1,id_label_err_lst[:,2].astype(float).max()+1])
    plt.xticks(np.linspace(-180,180,9,True))
    
    plt.savefig("MSE_error_wrt_label.png",bbox_inches='tight',pad_inches=0.2)

    ###############################################################################################################################
    # Plot MSE error wrt grid position (list)
    print("Save MSE wrt position plot")

    id_pos_err_lst = []     # make a list that has [id,grid_position,angle_MSE_error]
    label_angle = np.arctan2(np.array(labely_lst),np.array(labelx_lst))*180/np.pi      # degree
    output_angle = np.arctan2(np.array(outputy_lst),np.array(outputx_lst))*180/np.pi      # degree
    angle_error = np.abs(evaluation_error_deg)      # deg, this error returns high values since sometimes arctan2 returns e.g. 170deg for an angle of -180deg => very high error (prob same issue as single label)
    for i in range(len(id_lst)):
        pos = id_lst[i][-4:]
        if pos[0:3] == "000":
            pos = pos[-1]
        elif pos[0:2] == "00":
            pos = pos[-2:]
        elif pos[0] == "0":
            pos = pos[-3:]
        id_pos_err_lst.append([id_lst[i],pos,angle_error[i]])

    id_pos_err_lst = np.array(id_pos_err_lst)
    id_pos_err_lst = id_pos_err_lst[id_pos_err_lst[:,1].astype(float).argsort()]        # sort rows on the value of the pos

    # Plot settings
    fig = plt.figure(figsize=(10,10),dpi=300)
    plt.plot(id_pos_err_lst[:,1].astype(float),id_pos_err_lst[:,2].astype(float))
    plt.xlabel('grid position error [-]')
    plt.ylabel('error: abs(y-y*) [deg]')
    plt.xlim([id_pos_err_lst[:,1].astype(float).min()-1,id_pos_err_lst[:,1].astype(float).max()+1])
    plt.ylim([id_label_err_lst[:,2].astype(float).min()-1,id_label_err_lst[:,2].astype(float).max()+1])

    plt.savefig("MSE_error_wrt_pos.png",bbox_inches='tight',pad_inches=0.2)

    ###############################################################################################################################
    # for more control over mesh: https://stackoverflow.com/questions/44473531/i-am-plotting-a-3d-plot-and-i-want-the-colours-to-be-less-distinct
    # Plot 3D, MSE error wrt label and position
    print("Save 3D MSE plot wrt label and position")

    id_label_pos_err_lst = []       # make a 3d plot with (x,y,z)-axis = [id,label_angle,grid_position,error]
    label_angle = np.arctan2(np.array(labely_lst),np.array(labelx_lst))*180/np.pi      # degree
    output_angle = np.arctan2(np.array(outputy_lst),np.array(outputx_lst))*180/np.pi      # degree
    angle_error = np.abs(evaluation_error_deg)      # deg, this error returns high values since sometimes arctan2 returns e.g. 170deg for an angle of -180deg => very high error (prob same issue as single label)
    for i in range(len(id_lst)):
        pos = id_lst[i][-4:]
        if pos[0:3] == "000":
            pos = pos[-1]
        elif pos[0:2] == "00":
            pos = pos[-2:]
        elif pos[0] == "0":
            pos = pos[-3:]
        id_label_pos_err_lst.append([id_lst[i],label_angle[i],pos,angle_error[i]])

    id_label_pos_err_lst = np.array(id_label_pos_err_lst)
    id_label_pos_err_lst = id_label_pos_err_lst[id_label_pos_err_lst[:,1].astype(float).argsort()]

    # Plot settings
    fig = plt.figure(figsize=(10,10),dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(id_label_pos_err_lst[:,1].astype(float), id_label_pos_err_lst[:,2].astype(int), id_label_pos_err_lst[:,3].astype(float), cmap=cm.coolwarm)
    ax.view_init(elev=30., azim=-45, roll=0)
    ax.set_xlabel("label angle [deg]")
    ax.set_ylabel("grid position index [-]")
    ax.set_zlabel("error: abs(y-y*) [deg]")
    axins = inset_axes(ax,width="5%",height="80%",loc='center right',borderpad=-8)
    plt.colorbar(surf,cax=axins)

    plt.savefig("threeD_error_wrt_label_pos.png",bbox_inches='tight',pad_inches=0.2)