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
