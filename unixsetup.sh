#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    #Ubuntu
    wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
    chmod +x ./Anaconda3-2019.10-Linux-x86_64.sh
    ./Anaconda3-2019.10-Linux-x86_64.sh #install anaconda
    rm Anaconda3-2019.10-Linux-x86_64.sh

elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
	curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-MacOSX-x86_64.pkg
	chmod +x ./Anaconda3-2019.10-MacOSX-x86_64.pkg
	sudo installer -pkg Anaconda3-2019.10-MacOSX-x86_64.pkg -target /
	rm Anaconda3-2019.10-MacOSX-x86_64.pkg

fi

conda create --name tracking python==3.7.1
conda activate tracking
conda install bokeh==1.0.4
conda install -c menpo opencv
conda install -c anaconda tk
conda install -c anaconda cython scipy
conda install -c conda-forge matplotlib
cd cython_modules/
python setup.py build_ext --inplace
cd ../
python tutils.py
python guitest.py
