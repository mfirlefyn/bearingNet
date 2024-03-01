# BearingNet: Direct Learning of Home Vector Direction for Insect-inspired Robot Navigation

![alt text](https://github.com/mfirlefyn/bearingNet/blob/master/CNN.png "CNN image")

## Clean Installation Instructions

This guide is meant for bearingnet (approach from paper) and Flightmare installation on a fresh Ubuntu 20.04 machine. It is not excluded that the installation procedure will work on a not-so-fresh machine. It may also work on other Ubuntu or Linux flavors. This is left for the reader. 

Just follow along with the guide and copy paste the lines in the terminal after the previous command has finished. Whenever lines have to be pasted line-by-line in terminal, the instruction description will have a "(!)" indication. If the instruction lines can be copy pasted in the terminal as a whole, no further distinction is made.

### Cloning the Software
Open a terminal and let's get started. First of all, clone the bearingNet repository to your "/home" directory, rename it, and track the master branch (!):
```console
git clone https://github.com/mfirlefyn/bearingNet
mv -r bearingNet bearing_net
cd bearing_net
git checkout master
```

### Conda Virtual Environment Setup
Now we need to make sure that we do not mess up our inherent Python installation by congesting it with a bunch of packages that are barely used. The easiest way to keep everything clean is to use a virtual environment. You have several options to set up these, but we prefer using [conda](https://conda.io/projects/conda/en/latest/index.html) (!):
```console
cd ~/Downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the instructions on the shell. When it prompts you to press "enter" and when it gives you the long list with "--MORE--" you can hold the "f" key to scroll through it faster. It should prompt you whether you would want to accept the license terms. Type "yes" and press "enter", otherwise you can't use the software anyway. By pressing "enter" again at the install prompt, it installs miniconda in your "/home" directory, which is just fine for most use cases.

Finally, the terminal will prompt you whether you would want to initialize conda on every shell session in the future. Simply press "enter" to chose the default option "[no]". Otherwise, you start in a conda environment every single time you open a terminal. Not ideal.

Before conda will work, we will need to link the conda installation to our bash profile such that the computer knows where to look for next time we want to run something using conda. Make sure you replace "<user>" with your actual username on the system:
```console
echo ". /home/<user>/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
```

You only need to take this step if you messed up the previous step. Do not try to fill up your "~/.bashrc" file with nonsense by rerunning the same command. Just open up the bash profile settings in an editor of your choice and change the last line manually, save, and exit:
```console
gedit ~/.bashrc
```

Close your current terminal and open a new one. If you inspect your terminal it will now say "(base)" in the beginning of the line. That is exactly what we do not want. Thus, run the following:
```console
conda config --set auto_activate_base false
```

You could arrive to the same setup in a different way. If you inspect your "~/.bashrc" file again, you may notice it generated some extra instrcutions for conda. The installation procedure is a bit messy and not that straightforward, but all that matters is that you arrive at a point where you can open a normal terminal window again and that the "conda" command is recognized by the system. In other words, it should give you the conda usage explanation when you run in your current and subsequent new terminal windows:
```console
conda
```

Initialize conda manually in your current shell and make the virtual environment (!):
```console
conda init
conda create -n bnenv python=3.6 -y
```

You can check out all your environments by running:
```console
conda env list
```

To activate your environment in current or future terminal sessions:
```console
conda activate bnenv
```

Notice that now you can see the current environment in brackets "(bnenv)" at the beginning of the terminal line, just like when we were in the "(base)" environment. To exit your current environment and go back to a normal terminal session, type:
```console
conda deactivate
```

### Python Package Installation
This section will be quite repetitive. Unfrotunately, it's just the way it is. Once we get through this setup we can start on the interesting use cases of the software. So, let's get started setting up that environment of yours.

Be sure that you are currently in a terminal with a conda environment. Now, we need to make sure we have all the dependencies on our system to be able to run the software in this repository. We install most of the packages using the conda build-in package manager. Whenever it is not possible to use conda, we can use pip instead. 

Install the following default packages (!):
```console
conda activate bnenv
conda install numpy pandas scipy pyzmq -y
```

When installing third party packages from "conda-forge", we only have to state the package repository in the first command to add the "conda-forge" channel to our environment. Install the following third party packages (!):
```console
conda install conda-forge::matplotlib -y
conda install opencv ros-roslaunch ros-rospy -y
```

Lastly, we want to install PyTorch locally with their easy to use install command tool. We will only install PyTorch for use with the computer's cpu. For other installations using gpu acceleration, please consult the [PyTorch website](https://pytorch.org/get-started/locally/). Install pytorch for the cpu:
```console
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Extras Installation
We need to install earlyoom on your system to keep the RAM usage in check while running the simulation (!):
```console
sudo apt update
sudo apt install earlyoom -y
```

## Running the Program

We need to run several terminals at once in order to have a smooth user experience for the software. Terminals will be indicated by number, e.g. "terminal 1" with their explination in order to make the instructions as clear as possible. You can close any open terminals for now.

Open a terminal (terminal 1) and start up earlyoom:
```console
earlyoom
```

Open another terminal (terminal 2) and run roscore to keep track of all the open ros nodes (!):
```console
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
roscore
```

### Running the Offline Version of the Program
Depending on whether you want the training trajectory in a grid form or spiral form you need to invoke different versions of the program to run all the necessary parts. You will have to comment/uncomment the relevant code in our [Flightmare](https://github.com/mfirlefyn/flightmare) repo in order to simulate the correct training trajectory. You can also opt to copy paste the datasets manually and uncomment part of the offline scripts that handle Flightmare startup.

#### Rectangular Grid Training Trajectory
Open another terminal (terminal 3), get into your conda environment, source the ROS setup scripts and run the simulation (!):
```console
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
cd bearing_net
conda activate bnenv
python3 offline_rectGrid.py
```

It may happen that the script does not run due to not interpreting the ROS launch path correctly. You may have to open up the script in your favorite text editor and change the tilde expansion path "cli_args = ["~/catkin_ws/src/flightmare/flightros/launch/camera/camera.launch",'dist:='+str(dist)]" to an explicit path "cli_args = ["/home/<user>/catkin_ws/src/flightmare/flightros/launch/camera/camera.launch",'dist:='+str(dist)]", where "<user>" is replaced by your system's username.

Like in the Flightmare repo, it may be worthwile to note the PID of the "flightrender" and "camera" node since the renderer is quite unstable. If the renderer is only returning a blank screen, you may have to kill the application by invoking ```kill -9 <PID>``` in another terminal and replacing the "<PID>" field with the appropriate PID value.

#### Spiral Training Trajectory
Terminal 1 and 2 need to be opened in the same manner as described before. Terminal 3 can be opened the same way as in the rectangular grid case by replacing the Python script (!):
```console
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
cd bearing_net
conda activate bnenv
python3 offline_spiral.py
```

Again, be aware that the renderer can hang and needs to be killed and the terminal restarted in order to be able to run the program and generate your training images. Just keep iterating until the program runs. Occasionally, you may encounter a zmq error as well due to having a timeout of middleware communication somewhere if the program hangs too long. Again, just kill the program and reiteration can function as an effective work-around.

### Running the Online Version of the Program
If you want to generate the omni-directional images, fly the outbound trajectory, inbound trajectory, and evaluate the images while the robot is tracking its home vector continuously, this is the program you want to run.

Terminal 1 needs to be opened in the same manner as described before. Terminal 2 is not necessary in this case. Again, terminal 3 can be opened in much the same way as it was opened during the offline cases by replacing the Python script (!):
```console
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
cd bearing_net
conda activate bnenv
python3 offline_spiral.py
```

After running the script in terminal 3, you will be greeted by the "+++++WAIT FOR START TRAIN ZMQ MESSAGE+++++" message. It means that the script is waiting for the Flightmare program to be run and to send a message indicating that it has saved all the omni-directional training images. 

If you have not installed our [Flightmare software](https://github.com/mfirlefyn/flightmare), be sure to follow those install instructions before you attempt to continue. For now, another terminal (terminal 4) needs to be opened to run Flightmare (!):
```console
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
cd catkin_ws
roslaunch flightros camera.launch
```

After Flightmare has generated and stored the omni-directional images, it will send a zmq message to Python and the script will start training. Just wait for the script to finish its training cycle before you go into the evaluation flight phase of the Flightmare program, as explained in [its documentation](https://github.com/mfirlefyn/flightmare). 

Terminal 3 will indicate "print("+++++WAIT FOR EVALUATION ZMQ MESSAGE+++++")" while it is waiting for you to initialize the evaluation flight phase. Once initiated, the Flightmare component (terminal 4) and Python program (terminal 3) will work together to compute and display the effects of the estimated home vector on a view-per-view basis.

That's it. Hopefully you now know enough now to start your own journey with Flightmare, Python and ROS. Your simulation can take its virtual place now. Good luck and keep it adventurous!