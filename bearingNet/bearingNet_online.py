'''
Train and evaluate generated rectified catadioptric images (online) 
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

# data settings
batchSize = 1

# training dataset object definition
training_data = CustomImageDataset("labels.csv","img_dir")

# define NN modules
import torch.nn as nn
from torch import tanh

# define the CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 5, stride=4)
        self.conv2 = nn.Conv2d(2, 4, 5, stride=4)
        self.fc1 = nn.Linear(5376, 2)   # 9856 for real # 5376 for 201x1800 # 4752 for 201x1600
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
import time
import zmq

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
    
    #name = str(month_number) + "-" + first[0][3:] + "-" + first[1]

    return(month_number + day_number + hour_number + min_number + sec_number)

def getNameLatestRun(dir):
    names = os.listdir(dir)
    runs_number = [convertNameToNumber(name) for name in names]

    return(names[runs_number.index(max(runs_number))])

def getDirNameFromLatestRun(latest):
    name = latest.split("_")

    return(name[0] + "_" + name[1])

def getGazeAngleFromString(string):
    image_id = string.split("-")[-1].split(".")[0]
    if len(image_id) == 2:
        return(0)
    else:
        gaze_angle = np.linspace(-np.pi,np.pi,360,endpoint=False)
        idx = int(image_id[:-2])
        return(gaze_angle[idx]/np.pi/1.1)

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
    # loading best model and testing output
    net.load_state_dict(torch.load(evalPATH))

    # Take evaluation image from folder where Flightmare has saved it
    print("Evaluate image coming in from flightmare")

    evaluation_image = read_image("~/catkin_ws/src/flightmare/flightros/src/camera/eval_images/rectified-eval-img.jpg")
    inputs = evaluation_image.float()/255.
    inputs = inputs.unsqueeze(0)        # because of the absence of the dataloader, the batch needs to be accounted for 'artificially'
    outputs = net(inputs)
    
    # stdout some expressions to keep track of what Python state is to compare to Flightmare state
    print(f"outputs: {outputs[0,0].item()}, {outputs[0,1].item()}")
    print(f"distance of home vector: {np.sqrt(outputs[0,0].item()**2+outputs[0,1].item()**2)}")
    print(f"output angle: {np.arctan2(outputs[0,1].item(),outputs[0,0].item())*180/np.pi}")

    # send a zmq message to flightmare with the coords of the home vector
    context = zmq.Context() # ZeroMQ Context

    # Define the socket using the "Context"
    sock = context.socket(zmq.PUB)
    sock.connect("tcp://127.0.0.1:10256")

    # convert the outputs to a string
    output_message = str(outputs[0,0].item()) + ", " + str(outputs[0,1].item())

    # Send out the message with the evaluated output home vector, needs to be send twice (assuming that the zmq messaging only accepts after it has made contact), won't work correctly otherwise
    for i in range(2):
        sock.send_multipart([b"Home", output_message.encode("UTF-8"), str(i).encode("UTF-8")])  # b stands for byte buffer
        time.sleep(0.01)

    # close comm port after message has been
    sock.close()