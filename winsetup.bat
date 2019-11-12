ECHO Installing python libraries for particle tracking

conda create --name tracking python==3.7.1
conda activate tracking
conda install bokeh==1.0.4
conda install -c menpo opencv
conda install -c anaconda tk
conda install -c anaconda cython scipy
conda install -c conda-forge matplotlib
cd cython_modules/
python setup.py build_ext --inplace
cd ..
python tutils.py
python guitest.py