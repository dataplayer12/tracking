# Track Master 6000

A cross-platform particle tracking software.

<img src="https://github.com/dataplayer12/tracking/blob/master/images/tracklogo.png" width="64" height="128" />

# Contents
- [Citation](https://github.com/dataplayer12/tracking/blob/master/README.md#citation)
- [Abstract](https://github.com/dataplayer12/tracking/blob/master/README.md#abstract)
- [Setup instructions](https://github.com/dataplayer12/tracking/blob/master/README.md#setup)
- [Usage instructions](https://github.com/dataplayer12/tracking/blob/master/README.md#usage)
- [Differences between different releases](https://github.com/dataplayer12/tracking/blob/master/README.md#releases)
- [Naming](https://github.com/dataplayer12/tracking/blob/master/README.md#naming)

# Supported platforms
- Ubuntu
- macOS
- Windows

![Linux](https://github.com/gilbarbara/logos/blob/master/logos/linux-tux.svg)
<img src="https://github.com/gilbarbara/logos/blob/master/logos/macOS.svg" width="300" height="300" />
![windows](https://github.com/gilbarbara/logos/blob/master/logos/microsoft-windows.svg)

# Citation

If you use this software in your research, please cite us as
```
Jaiyam Sharma, Taisuke Ono, Adarsh Sandhu,
Smartphone enabled medical diagnostics by optically tracking electromagnetically induced harmonic oscillations of magnetic particles suspended in analytes,
Sensing and Bio-Sensing Research,
Volume 29,
2020,
100347,
ISSN 2214-1804,
https://doi.org/10.1016/j.sbsr.2020.100347.
(http://www.sciencedirect.com/science/article/pii/S2214180420300350)
```

# Abstract
The full paper can be found [here](https://www.sciencedirect.com/science/article/pii/S2214180420300350). The abstract is the following:
> Smartphone based point of care (POCT) medical diagnostic technology could potentially facilitate healthcare to remote locations with limited medical infrastrucure. However, such protocols exhibit background noise due to non-specific interactions as well as requiring long measurement times that limit their applications. We propose a procedure based on magnetic particle (MPs) that combines three dimensional electromagnetically induced oscillation of MPs with high precision optical tracking of the MPs. The measurement consisted of dropping MPs functionalized with streptavidin onto gold thin film actuators fabricated on silicon nitride substrates. Sinusoidal currents were passed through the actuators to move the MPs by horizontal dielectrophoretic forces. Simultaneously, a vertical magnetic field was applied to promote interaction between the MPs with biotin functionalized onto the substrates. The surfaces of MPs were functionalized with competing biotin whose concentration was varied. A high-resolution (4 K) video of the sensing surface was recorded with a smartphone and simultaneous tracking of 8000–10,000 MPs allowed us to identify MPs that interacted specifically with probes on the surface and ceased to show harmonic oscillations. Importantly, the dielectrophoretic forces reduced non-specific interactions and enhanced the probability of specific interactions while vertical magnetic forces accelerated the interaction of MPs with the functionalized surface. Our approach enabled a detection limit of 1 nM for biotin with a dynamic range of four orders of magnitude with a measurement time of approximately 2 min and video analysis time of 7 min. These results show that combining MPs and dynamic particle tracking is a promising method for practical POCT.

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

# Naming
This software is named `Track Master 6000`(TM6k). The name comes from [Dilbert](https://en.wikipedia.org/wiki/Dilbert), a comic series created by Scott Adams, one of my favorites. In the very [first episode](https://dilbert.fandom.com/wiki/The_Name) of the first season (of the animated television series based on the comic), Dilbert's boss decides to name a new product which has not been developed yet and for which no specifications have been decided. Dilbert's mom (Dilmom) suggests the name `GruntMaster 6000`, a name which gives the feeling that the product is less advanced than `GruntMaster 9000` (which also does not exist), but just as fun. This software was written in conjunction with experiments being done in our lab and its functionality was expanded and improved until we realized that it would be useful for other researchers working in particle tracking and biosensing. However, contrary to Dilbert, we had a product but no name. So, it felt fitting to name it `Track Master 6000`, after Dilmom's sage advice. TM6k is less advanced than a hypothetical TM9k, but just as useful. I hope you will find it useful in your research. Give it a try.
