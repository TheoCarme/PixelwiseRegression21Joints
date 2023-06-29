# Hands pose estimation on 3D camera using Pixel-wise Regression and Mediapipe Hands 

## Setup

### Nvidia things

Install the GPU driver that suits your configuration [here](https://www.nvidia.fr/Download/index.aspx?lang=fr)\
Then install CUDA 11.8 [here](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
Finally install cuDNN, following these [steps](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux) if you are on Linux and these [steps](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows) if you are on Windows.

### Python

Install a [version](https://www.python.org/downloads/) of Python equal or higher than 3.7\
Then install *pipenv* with this command : `pip install pipenv`\
You can now clone this repository with this command : `git clone https://github.com/TheoCarme/PixelwiseRegression21Joints.git`\

### ZED SDK

[Download](https://www.stereolabs.com/developers/release/) the installer for ZED SDK 4.0 that suits your OS. Then install it.

### Virtual environment

## Dataset and Models

If you intend to use test python program or to train the models yourself, then you will need to follow the following steps.
All datasets should be placed in `./Data` folder. After placing datasets correctly, run `python check_dataset.py --dataset <dataset_name>` to build the data files used to train.

### MSRA  
1. Download the dataset from [dropbox](https://www.dropbox.com/s/bmx2w0zbnyghtp7/cvpr15_MSRAHandGestureDB.zip?dl=0).
2. Unzip the files to `./Data` and rename the folder as `MSRA`.

### HAND17  
1. Ask for the permission from the [website](http://icvl.ee.ic.ac.uk/hands17/challenge/) and download.  
2. Download center files from github release, and put them in `Data/HAND17/`.
3. Extract `frame.zip` and `images.zip` to `./Data/HAND17/`. Your should end with a folder look like below:
```
HAND17/
  |
  |-- hands17_center_train.txt
  |
  |-- hands17_center_test.txt
  |
  |-- training/
  |     |
  |     |-- images/
  |     |
  |     |-- Training_Annotation.txt
  |
  |-- frame/
  |     |
  |     |-- images/
  |     |
  |     |-- BoundingBox.txt
```
### Pretrained models

You can also download the pretrained models [here](https://github.com/IcarusWizard/PixelwiseRegression/releases/tag/v1.0)
You can either download *HAND17_default_final.pt* or *MSRA_models.tar.gz* and place them in `./Model`
*MSRA_models.tar.gz* needs to be unzipped as it contains one model per subject in the dataset.

## Train  
Run `python train.py --dataset <dataset_name>`, `dataset_name` can be chose from `NYU`, `ICVL` and `HAND17`.  

For `MSRA` dataset, you should run `python train_msra.py --subject <subject_id>`.

## Test  
Run `python test.py --dataset <dataset_name>`.

For `MSRA` dataset, you should run `python test_msra.py --subject <subject_id>`.
