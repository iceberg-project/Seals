## Quality Metrics

[![Build Status](https://travis-ci.com/iceberg-project/Seals.svg?branch=devel)](https://travis-ci.com/iceberg-project/Seals)

## Software Dependencies


- torch 0.4.0
- torchvision 0.2.1
- tensorboardX 1.8
- cv2
- rasterio
- affine
- pandas
- geopandas
- numpy
- PIL


## Installation

### PSC Bridges
From source:
```bash
iparask$ module load cuda
iparask$ module load python3
iparask$ virtualenv iceberg_seals
iparask$ source iceberg_seals/bin/activate
[iceberg_seals] iparask$ export PYTHONPATH= <path>/iceberg_seals/bin/activate
[iceberg_seals] iparask$ pip install . --upgrade
```

From PyPi:
```bash
iparask$ module load cuda
iparask$ module load python3
iparask$ virtualenv iceberg_seals
iparask$ source iceberg_seals/bin/activate
[iceberg_seals] iparask$ export PYTHONPATH= <path>/iceberg_seals/bin/activate
[iceberg_seals] iparask$ pip install iceberg_seals.search
```

To test
```bash
[iceberg_seals] iparask$ iceberg_seals.tiling --scale_bands=299 --input_image=<image_abspath> --output_folder=./test
[iceberg_seals] iparask$ iceberg_seals.predicting --input_image=<image_filename> --model_architecture=UnetCntWRN --hyperparameter_set=A --training_set=test_vanilla --test_folder=./test --model_path=./ --output_folder=./test_image
```