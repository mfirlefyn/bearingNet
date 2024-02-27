# BearingNet: Direct Learning of Home Vector Direction for Insect-inspired Robot Navigation

### Conda Virtual Environment Setup
Open a terminal and let's get started. First of all, clone the bearingNet repository to your "/home" directory, rename it, and track the master branch (!):
```console
git clone https://github.com/mfirlefyn/bearingNet
mv -r bearingNet bearing_net
cd bearing_net
git checkout master
```

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
conda create -n bnenv -y
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
Be sure that you are currently in a terminal with a conda environment. Now, we need to make sure we have all the dependencies on our system to be able to run the software in this repository. We install most of the packages using the conda build-in package manager. Whenever it is not possible to use conda, we can use pip instead.

This section will be quite repetitive. Unfrotunately, it's just the way it is. Once we get through this setup we can start on the interesting use cases of the software. So, let's get started setting up that environment of yours.

