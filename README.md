## Quality Metrics

[![Build Status](https://travis-ci.com/iceberg-project/Seals.svg?branch=devel)](https://travis-ci.com/iceberg-project/Seals)

## Prerequisites - all available on bridges via the commands below
- Linux
- Python 3
- CPU and NVIDIA GPU + CUDA CuDNN

## Software Dependencies - these will be installed automatically with the installation below.
- numpy
- scipy
- pandas
- torch
- torchvision
- tensorboardX
- opencv-python
- rasterio
- affine
- geopandas
- pandas

## Installation
Preliminaries:  
These instructions are specific to XSEDE Bridges but other resources can be used if cuda, python3, and a NVIDIA P100 GPU are present, in which case 'module load' instructions can be skipped, which are specific to Bridges.  
  
For Unix or Mac Users:    
Login to bridges via ssh using a Unix or Mac command line terminal.  Login is available to bridges directly or through the XSEDE portal. Please see the [Bridges User's Guide](https://portal.xsede.org/psc-bridges).  

For Windows Users:  
Many tools are available for ssh access to bridges.  Please see [Ubuntu](https://ubuntu.com/tutorials/tutorial-ubuntu-on-windows#1-overview), [MobaXterm](https://mobaxterm.mobatek.net) or [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/)

### PSC Bridges 2
Once you have logged into bridges, you can follow one of two methods for installing iceberg-seals.

#### Method 1 (Recommended):  

The lines below following a '$' are commands to enter (or cut and paste) into your terminal (note that all commands are case-sensitive, meaning capital and lowercase letters are differentiated.)  Everything following '#' are comments to explain the reason for the command and should not be included in what you enter.  Lines that do not start with '$' or '[seals_env] $' are output you should expect to see.

```bash
$ pwd
/home/username
$ cd $SCRATCH                      # switch to your working space.
$ mkdir Seals                      # create a directory to work in.
$ cd Seals                         # move into your working directory.
$ module load cuda/10.2.0          # load parallel computing architecture.
$ module load anaconda3            # load correct python version.
$ conda create -p seals_env python=3.9 -y     # create a virtual environment to isolate your work from the default system.
$ source activate <path>/seals_env    # activate your environment. Notice the command line prompt changes to show your environment on the next line.
[seals_env] $ pwd
/pylon5/group/username/Seals
[seals_env] $ export PYTHONPATH=<path>/seals_env/lib/python3.9/site-packages # set a system variable to point python to your specific code. (Replace <path> with the results of pwd command above.
[seals_env] $ pip install iceberg_seals.search # pip is a python tool to extract the requested software (iceberg_seals.search in this case) from a repository. (this may take several minutes).
```

#### Method 2 (Installing from source; recommended for developers only): 

```bash
$ git clone https://github.com/iceberg-project/Seals.git
$ module load cuda/10.2.0
$ module load anaconda3              # load correct python version.
$ conda create -p seals_env python=3.9 -y     # create a virtual environment to isolate your work from the default system.
$ source activate <path>/seals_env    # activate your environment. Notice the command line prompt changes to show your environment on the next line.
[seals_env] $ export PYTHONPATH=<path>/seals_env/lib/python3.9/site-packages # set a system variable to point python to your specific code. (Replace <path> with the results of pwd command above.
[seals_env] $ pip install . --upgrade
```

#### To test
```bash
[iceberg_seals] $ deactivate       # exit your virtual environment.
$ interact --gpu  # request a compute node.  This package has been tested on P100 GPUs on bridges, but that does not exclude any other resource that offers the same GPUs. (this may take a minute or two or more to receive an allocation).
$ cd $PROJECT/Seals                # make sure you are in the same directory where everything was set up before.
$ module load cuda/10.2.0          # load parallel computing architecture, as before.
$ module load anaconda3            # load correct python version, as before.
$ source activate <path>/seals_env    # activate your environment. Notice the command line prompt changes to show your environment on the next line.
[seals_env] $ export PYTHONPATH=<path>/seals_env/lib/python3.9/site-packages # set a system variable to point python to your specific code. (Replace <path> with the results of pwd command above.
[iceberg_seals] $ iceberg_seals.predicting --help  # this will display a help screen of available usage and parameters.
```
## Prediction
- Download a pre-trained model at: https://github.com/iceberg-project/Seals/tree/master/models/Heatmap-Cnt/UnetCntWRN/UnetCntWRN_ts-vanilla.tar 

You can download to your local machine and use scp, ftp, rsync, or Globus to [transfer to bridges](https://portal.xsede.org/psc-bridges).

Seals predicting is executed in two steps: 
First, follow the environment setup commands under 'To test' above. Then create tiles from an input GeoTiff image and write to the output_folder. The scale_bands parameter (in pixels) depends on the trained model being used.  The default scale_bands is 224 for the pre-trained model downloaded above.  If you use your own model the scale_bands may be different.
```bash
[iceberg_seals] $ iceberg_seals.tiling --scale_bands=224 --input_image=<image_abspath> --output_folder=./test
```
Then, detect seals on each tile and output counts and confidence for each tile.
```bash
[iceberg_seals] $ iceberg_seals.predicting --input_image=<image_filename> --model_architecture=UnetCntWRN --hyperparameter_set=A --training_set=test_vanilla --test_folder=./test --model_path=./ --output_folder=./test_image
```
