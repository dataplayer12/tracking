# tracking
Particle tracking software


## Setup

```Shell
git clone http://www.github.com/dataplayer12/tracking.git
cd tracking
conda create --name tracking --file environment.txt
conda activate tracking
cd cython_hw/
python setup.py build_ext --inplace
cd ../
python tutils.py
python guitest.py
```
After the last command, a new window should open. It should look like this:
![guitest](https://github.com/dataplayer12/tracking/blob/master/guitest.png)

If you can successfully see the text 'Hello World' in the window, the installation was successful. Now, use the `frontend` script as described below.

## Usage

Within the tracking environment, as setup before, use the `frontend` script as
```Shell
python frontend.py
```
This will bring up a graphical user interface as below:
![gui](https://github.com/dataplayer12/tracking/blob/master/gui.png)

The GUI is easy to use and is still under some development. Basic tracking functionality with a single process and thread is available int he current version.

During tracking the software displays a progress bar:
![waitbar](https://github.com/dataplayer12/tracking/blob/master/waitbar.png)
