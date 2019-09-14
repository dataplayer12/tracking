# tracking
Particle tracking software
![logo](https://github.com/dataplayer12/tracking/blob/master/images/tracklogo.png)

## Supported platforms
![macos](https://github.com/gilbarbara/logos/blob/master/logos/macosx.svg)
![Linux](https://github.com/gilbarbara/logos/blob/master/logos/linux-tux.svg)
![windows](https://github.com/gilbarbara/logos/blob/master/logos/microsoft-windows.svg)

## Setup

```Shell
git clone http://www.github.com/dataplayer12/tracking.git
cd tracking
conda create --name tracking --file environment.txt
conda activate tracking
cd cython_modules/
python setup.py build_ext --inplace
cd ../
python tutils.py
python guitest.py
```
After the last command, a new window should open. It should look like this:
![guitest](https://github.com/dataplayer12/tracking/blob/master/images/guitest.png)

If you can successfully see the text 'Hello World' in the window, the installation was successful. Now, use the `frontend` script as described below.

## Usage

Within the tracking environment, as setup before, use the `frontend` script as
```Shell
python frontend.py
```
This will bring up a graphical user interface as below:
![gui](https://github.com/dataplayer12/tracking/blob/master/images/gui.png)

The GUI is easy to use and is still under some development.

During tracking the software displays a progress bar:
![waitbar](https://github.com/dataplayer12/tracking/blob/master/images/waitbar.png)

## Release 0.2
Version 0.2 of TM6000 introduces modes for easier setting of two kinds of experiments. This release also introduces multi-processing for tracking several videos in parallel on a multi-core computer. For example, here we show tracking of 15 videos simultaneously on a Ryzen 3700x system running Ubuntu 18.04.

![multi-processing](https://github.com/dataplayer12/tracking/blob/master/images/multi-processing.png)
