## Quality Metrics

[![Build Status](https://travis-ci.com/iceberg-project/Seals.svg?branch=devel)](https://travis-ci.com/iceberg-project/Seals)

## Prerequisites - all available on bridges via the commands below
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Software Dependencies - these will be installed automatically with the installation below.
- numpy
- scipy
- pandas
- torch==0.4.0
- torchvision==0.2.1
- tensorboardX==1.8
- opencv-python
- rasterio
- affine
- geopandas
- pandas

## Installation
Preliminaries:
Login to Bridges via ssh using a Unix or Mac command line terminal.  Login is available to Bridges directly or through the XSEDE portal. Please see the <a href="https://portal.xsede.org/psc-bridges">Bridges User's Guide</a>.  

For Windows Users:  
Many tools are available for ssh access to Bridges.  Please see <a href="https://ubuntu.com/tutorials/tutorial-ubuntu-on-windows#1-overview">Ubuntu</a>, <a href="https://mobaxterm.mobatek.net/">MobaXterm</a>, or <a href="https://www.chiark.greenend.org.uk/~sgtatham/putty/">PuTTY</a>

### PSC Bridges
Once you have logged into Bridges, you can follow one of two methods for installing iceberg-seals.

Method #1 (Recommended):  

(Note: The lines below starting with '$' are commands to type into your terminal.  Everything following '#' are comments to explain the reason for the command and should not be included in what you type.  Lines that do not start with '$' or '[seals_env] $' are output you should expect to see.)

```bash
$ pwd
/home/username
$ cd $SCRATCH                      # switch to your working space.
$ mkdir Seals                      # create a directory to work in.
$ cd Seals                         # move into your working directory.
$ module load cuda                 # load parallel computing architecture.
$ module load python3              # load correct python version.
$ virtualenv seals_env             # create a virtual environment to isolate your work from the default system.
$ source seals_env/bin/activate    # activate your environment. Notice the command line prompt changes to show your environment on the next line.
[seals_env] $ pwd
/pylon5/group/username/Seals
[seals_env] $ export PYTHONPATH=<path>/seals_env/lib/python3.5/site-packages # set a system variable to point python to your specific code. (Replace <path> with the results of pwd command above.
[seals_env] $ pip install iceberg_seals.search # pip is a python tool to extract the requested software (iceberg_seals.search in this case) from a repository. (this may take several minutes).
```

Method #2 (Installing from source; recommended for developers only): 

```bash
$ git clone https://github.com/iceberg-project/Seals.git
$ module load cuda
$ module load python3
$ virtualenv seals_env
$ source seals_env/bin/activate
[seals_env] $ export PYTHONPATH=<path>/seals_env/lib/python3.5/site-packages
[seals_env] $ pip install . --upgrade
```

To test
```bash
[iceberg_seals] $ deactivate    # exit your virtual environment.
$ interact -p GPU-small            # request a compute node (this may take a minute or two or more).
$ cd $SCRATCH/Seals             # make sure you are in the same directory where everything was set up before.
$ module load cuda                 # load parallel computing architecture, as before.
$ module load python3              # load correct python version, as before.
$ source seals_env/bin/activate # activate your environment, no need to create a new environment because the Seals tools are installed and isolated here.
[iceberg_seals] $ iceberg_seals.detect --help  # this will display a help screen of available usage and parameters.
```
## Prediction
- Download a pre-trained model at: https://github.com/iceberg-project/Seals/tree/feature/README/models/Heatmap-Cnt/UnetCntWRN/UnetCntWRN_ts-vanilla.tar 

You can download to your local machine and use scp, ftp, rsync, or Globus to transfer to bridges.

Seals predicting is executed in two steps:  
First, create tiles from an input GeoTiff image and write to the output_folder. The scale_bands parameter depends on trained model being used.  The default scale_bands is 299 for the pre-trained model above.
```bash
[iceberg_seals] $ iceberg_seals.tiling --scale_bands=299 --input_image=<image_abspath> --output_folder=./test
```
Then, detect seals on each tile and output counts and confidence for each tile.
```bash
[iceberg_seals] $ iceberg_seals.predicting --input_folder=./test --model_path=./<path_to_model> --output_folder=./test_image
```
