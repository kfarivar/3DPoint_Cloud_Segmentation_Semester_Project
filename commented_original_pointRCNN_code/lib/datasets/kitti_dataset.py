import os
import numpy as np
import torch.utils.data as torch_data
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
from PIL import Image


class KittiDataset(torch_data.Dataset):
        ''' is a reader that reads the files one by one.
        It is used in both generate_gt_database.py and kitti_rcnn_dataset.py
        
        This uses The text files containing ids in ImageSets 
        to group data for test/train/eval/...'''

    def __init__(self, root_dir, split='train'): #root dir ../data/
        self.split = split       
        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'KITTI', 'object', 'testing' if is_test else 'training')

        # it seems there is this default scene ids used as test/train
        #  
        # this line reads the file that contains the indexes for the test/train dataset
        split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', split + '.txt')
        #and they are put in the image_idx_list
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_sample = self.image_idx_list.__len__()

        # these are the paths for
        #images
        self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        #point clouds
        self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        # clibration 
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        #labels
        self.label_dir = os.path.join(self.imageset_dir, 'label_2')
        #road planes
        self.plane_dir = os.path.join(self.imageset_dir, 'planes')

    def get_image(self, idx):
        assert False, 'DO NOT USE cv2 NOW, AVOID DEADLOCK'
        import cv2
        # cv2.setNumThreads(0)  # for solving deadlock when switching epoch
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4) # the shape is (n, 4)
        # the data is x,y,z,r (r is intensity)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def get_road_plane(self, idx):

        '''example of the plane file:
        # Matrix
        WIDTH 4
        HEIGHT 1
        -1.851372e-02 -9.998285e-01 -5.362325e-04 1.678761e+00 '''
        # how does she get the road planes ?

        plane_file = os.path.join(self.plane_dir, '%06d.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()

        # 4 float values in line 4 of each file (what are they?)
        lines = [float(i) for i in lines[3].split()] 
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
