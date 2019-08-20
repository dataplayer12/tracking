# tracking
Particle tracking software


## Setup

```Shell
conda create -n tracking python=3.7
conda activate tracking
git clone http://www/github.com/dataplayer12/tracking
cd tracking
pip install -r requirements.txt
conda install -c menpo opencv
cd cython_hw/
python setup.py build_ext --inplace
cd ../
python tutils.py
```
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