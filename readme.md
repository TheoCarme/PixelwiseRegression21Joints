# Pixel-wise Regression for 3D hand pose estimation 
**PyTroch release of our paper:**   
[Pixel-wise Regression: 3D Hand Pose Estimation via Spatial-form Representation and Differentiable Decoder](https://arxiv.org/abs/1905.02085)  
*Xingyuan Zhang, Fuhai Zhang*

If you find this repository useful, please make a reference in your paper:
```
@misc{zhang2019pixelwise,
    title={Pixel-wise Regression: 3D Hand Pose Estimation via Spatial-form Representation and Differentiable Decoder},
    author={Xingyuan Zhang and Fuhai Zhang},
    year={2019},
    eprint={1905.02085},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Setup
```
conda env create -f env.yml
conda activate pixelwise
```

## Dataset  
All datasets should be placed in `./Data` folder. After placing datasets correctly, run `python check_dataset.py --dataset <dataset_name>` to build the data files used to train.

### NYU  
1. Download the dataset from [website](https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm#download).
2. Unzip the files to `./Data` and rename the folder as `NYU`.

### MSRA  
1. Download the dataset from [dropbox](https://www.dropbox.com/s/bmx2w0zbnyghtp7/cvpr15_MSRAHandGestureDB.zip?dl=0).
2. Unzip the files to `./Data` and rename the folder as `MSRA`.

### ICVL  
1. Download the dataset from [here](https://labicvl.github.io/hand.html).  
2. Extract `Training.tar.gz` and `Testing.tar.gz` to `./Data/ICVL/Training` and `./Data/ICVL/Testing` respectively.

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

## Train  
Run `python train.py --dataset <dataset_name>`, `dataset_name` can be chose from `NYU`, `ICVL` and `HAND17`.  

For `MSRA` dataset, you should run `python train_msra.py --subject <subject_id>`.

## Test  
Run `python test.py --dataset <dataset_name>`.

For `MSRA` dataset, you should run `python test_msra.py --subject <subject_id>`.

## Results  
Results and pretrained models are available in github release.