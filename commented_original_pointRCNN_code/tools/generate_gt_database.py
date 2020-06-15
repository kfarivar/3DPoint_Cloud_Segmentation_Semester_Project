import _init_path
import os
import numpy as np
import pickle
import torch

import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.datasets.kitti_dataset import KittiDataset
import argparse

#go here to see what argpars is all about
#https://docs.python.org/3/howto/argparse.html

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./gt_database')
parser.add_argument('--class_name', type=str, default='Car')
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()


class GTDatabaseGenerator(KittiDataset):
''' Ground truth database generator:
    The final result of this function is a list of objects. each object is defined as a dictionary of attributes :

        {'sample_id': sample_id, # scene id
            'cls_type': obj_list[k].cls_type, # class (e.g. Car)
            'gt_box3d': gt_boxes3d[k], # bounding box
            'points': cur_pts, # point coords
            'intensity': cur_pts_intensity,
            'obj': obj_list[k]}

    That is going to be used for augmentation.
    the results are saved in a binary pickle file. 
 '''

    ''' This is a data reader class for the whole data
    KittiDataset is also a reader but it reads the files one by one '''


    def __init__(self, root_dir, split='train', classes=args.class_name):
        super().__init__(root_dir, split=split)
        self.gt_database = None
        if classes == 'Car':
            self.classes = ('Background', 'Car')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif classes == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
        elif classes == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
        else:
            assert False, "Invalid classes: %s" % classes

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def filtrate_objects(self, obj_list):
        '''
        Only keep labeled data for classes that we want and the level of difficulty should not be more than hard (based on object size)
        I don't think including Background in 'self.classes' affects anything
        since it doesn't exist in the types that is given labes file 
        '''
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in self.classes:
                continue
            if obj.level_str not in ['Easy', 'Moderate', 'Hard']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    def generate_gt_database(self):
        gt_database = []

        # read (have it as a reference) the KITTI dataset documentation. 
        # it is a text file in the library they have for handeling data. 
        # Download object development kit (1 MB) (including 3D object detection and bird's eye view evaluation code)
        # http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

        # image_idx_list was defined in the parent
        # it is a list of ids for data to choose for test/train
        for idx, sample_id in enumerate(self.image_idx_list):
            # get the id of the image/pointcloud (corresponding to 1 scene)
            sample_id = int(sample_id)
            print('process gt sample (id=%06d)' % sample_id)

            # for later: There is the same number of images and point clouds so I think it is 1 image per point cloud


            # read point cloud data
            pts_lidar = self.get_lidar(sample_id)
            calib = self.get_calib(sample_id)
            # coordinate calibration on x,y,z (not intensity!) (rect stands for recalibrated ?!)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            # store intensity seperately 
            pts_intensity = pts_lidar[:, 3]

            # first get a list of objects in the scene using the label data 
            # then only pick (filter) those that are in the  self.classes
            obj_list = self.filtrate_objects(self.get_label(sample_id))

            # for each object (e.g. car) save the center (x,y,z) dimensions (w,h,l) and angle in the x-z plane (see paper figure3)
            gt_boxes3d = np.zeros((obj_list.__len__(), 7), dtype=np.float32)

            for k, obj in enumerate(obj_list):
                gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
                    = obj.pos, obj.h, obj.w, obj.l, obj.ry

            # if no objects of interest go to the next scene
            if gt_boxes3d.__len__() == 0:
                print('No gt object')
                continue

            # send the whole scene to 'pts_in_boxes3d_cpu' and the boxes in the scene
            # determin which points are inside the box the returned value is a 2d array 
            # each row i is a mask of points that belong the box i and it is a pytorch tensor
            boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(torch.from_numpy(pts_rect), torch.from_numpy(gt_boxes3d))

            # for each bounding box
            # choose the points inside a bounding box and label them as a single instance
            for k in range(boxes_pts_mask_list.__len__()):
                # 'boxes_pts_mask_list[k]' is a pytorch tensor
                pt_mask_flag = (boxes_pts_mask_list[k].numpy() == 1)
                
                #choose points that are inside the current box ('pts_rect' is a numpy array)
                cur_pts = pts_rect[pt_mask_flag].astype(np.float32)
                cur_pts_intensity = pts_intensity[pt_mask_flag].astype(np.float32)

                sample_dict = {'sample_id': sample_id, # scene id
                               'cls_type': obj_list[k].cls_type, # class (e.g. Car)
                               'gt_box3d': gt_boxes3d[k], # bounding box
                               'points': cur_pts, # point coords
                               'intensity': cur_pts_intensity,
                               'obj': obj_list[k]} # this includes all the other details we don't have above (as an object with attributes)
                gt_database.append(sample_dict)

        save_file_name = os.path.join(args.save_dir, '%s_gt_database_3level_%s.pkl' % (args.split, self.classes[-1]))
        with open(save_file_name, 'wb') as f:
            pickle.dump(gt_database, f)

        self.gt_database = gt_database
        print('Save refine training sample info file to %s' % save_file_name)


if __name__ == '__main__':
    # read the data from the root_dir
    dataset = GTDatabaseGenerator(root_dir='../../data/', split=args.split)
    
    # make folder to save the data
    os.makedirs(args.save_dir, exist_ok=True)

    dataset.generate_gt_database()

    # gt_database = pickle.load(open('gt_database/train_gt_database.pkl', 'rb'))
    # print(gt_database.__len__())
    # import pdb
    # pdb.set_trace()
