# Pipe segmentation

![teaser](https://github.com/KiyarashFarivar/3DPoint_Cloud_Segmentation_Semester_Project/blob/master/pipe_segmentation/rpn_network.png)

## Introduction
This was an attempt to apply the method suggested in the [PointRCNN Paper](https://arxiv.org/abs/1812.04244) ([repo](https://github.com/sshaoshuai/PointRCNN)) to the pipes dataset provided (dataset only available through the epfl CVLAB). Only the first stage network was used. For more info refer to the [report of the project](https://github.com/KiyarashFarivar/3DPoint_Cloud_Segmentation_Semester_Project/blob/master/Project_Report_and_Presentation/kiarash_farivar_semester_project_report.pdf). It is suggested to read the PointRCNN Paper before doing anything else. 

## Installation
### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 18.04.3 LTS)
* Python 3.7.4+
* PyTorch 1.2.0

### Install PointRCNN 

a. Clone the whole repository.

b. Install the dependent python libraries like `easydict`,`tqdm`, `tensorboardX ` etc.
```shell
conda create -n myenv python=3.7.4
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch 
conda install scipy numba shapely
conda install -c conda-forge easydict tqdm tensorboardx fire
pip install scikit-image pyyaml
```

c. cd into pipe_segmentation folder. Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:
```shell
sh build_and_install.sh
```

## Dataset preparation
Organize the files as follows (you can also make symbolic links in ubuntu instead of actually copying the data): 
```
pipe_segmentation
├-- data
│   |-- bbox_data (the resulting bboxes and points from the finding_bboxes.ipynb in data_cleaning_and_preprocessing folder)
│   |-- lidar_sampels (the result of two notebooks in lidar_sampling folder)
|   |-- original_data (the original data)
|   |-- final_cleaned_data (will be created after running the preprocess_data.py in tools folder)
├── lib
├── pointnet2_lib
├── tools
```
In all the following make sure you are in the tools folder.

## Training
To preprocess the data in the bbox_data, lidar_sampels and original_data run preprocess_data.py
```
python preprocess_data.py
```

### Training of RPN stage
* To train run the following command. --train_with_eval option can be used to evaluate the chckpoints as well; this gives a train_val_losses.npz file in the tools folder that has recorder the train and validation losses:
```
python train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 16 --train_mode rpn --epochs 200 --train_with_eval
```

After training, the checkpoints and training logs will be saved to the corresponding directory according to the name of your configuration file. Such as for the `default.yaml`, you could find the checkpoints and logs in the following directory:
```
PointRCNN/output/rpn/default/
```

## Inference \ Validation
* To evaluate a single checkpoint, run the following command with `--ckpt` to specify the checkpoint to be evaluated:
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt ../output/rpn/default/ckpt/checkpoint_epoch_10.pth --batch_size 4 --eval_mode rpn
```

* To evaluate all the checkpoints of a specific training config file, add the `--eval_all` argument, and run the command as follows. This gives a performances.npz file whith all checkpoints' performances:
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --eval_mode rpn --eval_all
```

## A short discription of which file is what
The central functions are in the tools folder. Most of the preprocessing and data cleaning happens in **preprocess_data.py**.

As mentioned above the main script for training is **train_rcnn.py** here we define a dataset in **create_dataloader(logger)** function. the **PointRCNN()** class creates the network. Finally **train_utils.Trainer** is where training happens. The loss function is defined separately in **train_functions.model_joint_fn_decorator()**. 

Use a text editor like vscode so you can follow the reference links to the definitions of classes and functions easily. I have tried my best to comment the code. For better understanding also refer to the commented version of the original code I have provided. 
