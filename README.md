# Track Master 6000

A cross-platform particle tracking software.

<img src="https://github.com/dataplayer12/tracking/blob/master/images/tracklogo.png" width="64" height="128" />

# Contents
- [Setup instructions](https://github.com/dataplayer12/tracking/blob/master/README.md#setup)
- [Usage instructions](https://github.com/dataplayer12/tracking/blob/master/README.md#usage)
- [Differences between different releases](https://github.com/dataplayer12/tracking/blob/master/README.md#releases)
- [Naming]()

# Supported platforms
- Ubuntu
- macOS
- Windows

![Linux](https://github.com/gilbarbara/logos/blob/master/logos/linux-tux.svg)
<img src="https://github.com/gilbarbara/logos/blob/master/logos/macOS.svg" width="300" height="300" />
![windows](https://github.com/gilbarbara/logos/blob/master/logos/microsoft-windows.svg)

# Setup
On **Windows**:
- [Download](https://www.anaconda.com/distribution/) and install the Anaconda python package manager.
- In the `Start` menu, search for `Anaconda Prompt` and open it.
- Click the `Clone or Download` button above and click `Download zip`.
- Unzip the folder and open it. You will find a file called `winsetup.bat`
- From `Anaconda Prompt`, go to the directory where the zip file was extracted and type `winsetup.bat` and press `Enter`.
- Please wait while the dependencies are being installed.

On **Linux or Mac**, open up the terminal and run the following commands:
```Shell
git clone http://www.github.com/dataplayer12/tracking.git
cd tracking
git checkout v0.3
sudo chmod +x ./unixsetup.sh #You will have to provide your password to run this command
sudo ./unixsetup.sh #This command will take a long time to execute.
```
If everything goes well, you will have the libraries installed on Linux/mac/Windows and a new window will open up. It will look like this:
![guitest](https://github.com/dataplayer12/tracking/blob/master/images/guitest.png)

If you can successfully see the text 'Hello World' in the window, the installation was successful. Now, use the `frontend` script as described below.

# Usage

Within the tracking environment, as setup before, use the `frontend` script as
```Shell
python frontend.py
```
This will bring up a graphical user interface as below:
![gui](https://github.com/dataplayer12/tracking/blob/master/images/gui.png)

The GUI is easy to use and is still under some development.

During tracking the software displays a progress bar:
![waitbar](https://github.com/dataplayer12/tracking/blob/master/images/waitbar.png)

# Releases

## Release 0.3
Version 0.3 of TM6k introduces a cloud daemon to which can analyze the files uploaded to a server on the cloud. The server can be started by invoking the server as `python cloudd.py`. While the server is running, this banner is displayed on the screen:
![cloudd](https://github.com/dataplayer12/tracking/blob/master/images/cloudlogo.png)
Press 'q' to stop the server.

## Release 0.2
Version 0.2 of TM6000 introduces modes for easier setting of two kinds of experiments. This release also introduces multi-processing for tracking several videos in parallel on a multi-core computer. For example, here we show tracking of 15 videos simultaneously on a Ryzen 3700x system running Ubuntu 18.04.

![multi-processing](https://github.com/dataplayer12/tracking/blob/master/images/multi-processing.png)

## Release 0.1
Version 0.1 of TM6000 is the first stable release of this software, introducing a GUI for particle tracking. Only one video can be tracked at a time.
