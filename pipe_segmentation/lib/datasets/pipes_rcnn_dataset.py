import numpy as np
from numpy import random
import os
import pickle
import torch
import sys

sys.path.insert(1, '/cvlabdata2/home/farivar/my_network/pipe_detection')

import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.config import cfg

from lib.datasets.pipes_dataset import pipes_dataset
import lib.utils.pipes_bbox_utils as pipe_utils


class pipes_rcnn_dataset(pipes_dataset):
    def __init__(self, root_dir, npoints=16384, split='train', classes='Pipe', mode='TRAIN', random_select=True,
                 logger=None, rcnn_training_roi_dir=None, rcnn_training_feature_dir=None, rcnn_eval_roi_dir=None,
                 rcnn_eval_feature_dir=None, gt_database_dir=None):

        super().__init__(root_dir=root_dir, split=split)

        # define the different classes we have 
        '''    
        0 Unclassified
        1 forestay
        2 valve
        3 Pipe
        4 T fitting
        5 reduction
        6 Y connector
        7 hydrating_standard
        8 fitting
        9 elbow
        10 cap
        '''
        if classes == 'Pipe':
            self.classes = ('Background', 'Pipe')
        elif classes == 'All':
            assert False, 'Not implemented'
        else:
            assert False, "Invalid classes: %s" % classes

        self.num_class = self.classes.__len__()

        # npoints is the number of points each of our scenes will be made of AFTER we have done 
        # preprocessing the data. This is important since pointnet++ maps the features back to each point. 
        # see the function "get_rpn_sample" in there if the scene has too much points we sample from the near points
        self.npoints = npoints

        # sample_id_list is a translator. we get an index in _getitem_(index) function but the real index 
        # of the scene is sample_id_list[index] 
        self.sample_id_list = []
        self.random_select = random_select
        self.logger = logger

        

        # for rcnn training
        '''self.rcnn_training_bbox_list = []
        self.rpn_feature_list = {}
        self.pos_bbox_list = []
        self.neg_bbox_list = []
        self.far_neg_bbox_list = []
        self.rcnn_eval_roi_dir = rcnn_eval_roi_dir
        self.rcnn_eval_feature_dir = rcnn_eval_feature_dir
        self.rcnn_training_roi_dir = rcnn_training_roi_dir
        self.rcnn_training_feature_dir = rcnn_training_feature_dir'''

        if not self.random_select:
            self.logger.warning('random select is False')

        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode


        if cfg.RPN.ENABLED:
            if mode == 'TRAIN':
                self.preprocess_rpn_training_data()
            else:
                self.sample_id_list = list(range(self.num_sample))
                self.logger.info('Done: total test samples %d' % len(self.sample_id_list))
        elif cfg.RCNN.ENABLED:
            for idx in range(0, self.num_sample):
                sample_id = int(self.image_idx_list[idx])
                obj_list = self.filtrate_objects(self.get_label(sample_id))
                if len(obj_list) == 0:
                    # logger.info('No gt classes: %06d' % sample_id)
                    continue
                self.sample_id_list.append(sample_id)

            print('Done: filter %s results for rcnn training: %d / %d\n' %
                  (self.mode, len(self.sample_id_list), len(self.image_idx_list)))

    def preprocess_rpn_training_data(self):
        """
        Discard samples which don't have current classes, which will not be used for training.
        Valid sample_ids are stored in self.sample_id_list
        """
        
        if self.classes ==('Background', 'Pipe'):
            # nothing to do all our samples already have pipes
            self.sample_id_list = list(range(self.num_sample))
        else:
            # if you want to select a subset put the index of the folders that you are interested in
            # in self.sample_id_list
            raise NotImplementedError





    def filtrate_objects(self, bboxes):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """

        # look at the keys in the bbox dictionary 
        # to select only the classes you need.

        # not implemented for specific classes yet
        valid_bbox_list = bboxes 

        return valid_bbox_list

    

    def __len__(self):
        if cfg.RPN.ENABLED:
            return len(self.sample_id_list)
        elif cfg.RCNN.ENABLED:
            if self.mode == 'TRAIN':
                return len(self.sample_id_list)
            else:
                return len(self.image_idx_list)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if cfg.RPN.ENABLED:
            return self.get_rpn_sample(index)
        elif cfg.RCNN.ENABLED:
            if self.mode == 'TRAIN':
                if cfg.RCNN.ROI_SAMPLE_JIT:
                    return self.get_rcnn_sample_jit(index)
                else:
                    return self.get_rcnn_training_sample_batch(index)
            else:
                return self.get_proposal_from_file(index)
        else:
            raise NotImplementedError

    def get_rpn_sample(self, index):
        
        # translate index
        sample_id = int(self.sample_id_list[index])
    
        # get all the info of the sample lidar scene
        pts_data, sample_bboxes = self.get_points_and_bbox(sample_id)

        # we don't have intensities (note even in the original code althought they do read the intensities it is never used in the RPN network)
        pts_intensity = np.zeros_like(pts_data[:,0])

        # generate inputs
        # fit the number of points to a specific size self.
        # (if running the rcnn script random_select is true so we enter the if)
        if self.mode == 'TRAIN' or self.random_select:
            # if too many points
            if self.npoints < len(pts_data):
                # calculate the euclidian norm of each point as depth
                # since we have normalized most points will have depth less than 2
                pts_depth = np.linalg.norm(pts_data[:,0:3], axis=1)
                pts_near_flag = pts_depth <= 1000.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                assert len(far_idxs_choice) < self.npoints
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)

                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(pts_data), dtype=np.int32)
                if self.npoints > len(pts_data):
                    # if scene has too few points do it with replacement
                    if self.npoints > 2*len(pts_data):
                        extra_choice = np.random.choice(choice, self.npoints - len(pts_data), replace=True)
                    else : 
                        extra_choice = np.random.choice(choice, self.npoints - len(pts_data), replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            chosen_pts = pts_data[choice, :]
            ret_pts_intensity = pts_intensity[choice] #- 0.5  # translate intensity to [-0.5, 0.5]
        else:
            chosen_pts = pts_data
            ret_pts_intensity = pts_intensity #- 0.5

        ret_pts_features = ret_pts_intensity.reshape(-1, 1)

        sample_info = {'sample_id': sample_id, 'random_select': self.random_select}

        # when running the eval_rcnn script mode == 'EVAL'
        if self.mode == 'TEST':
            if cfg.RPN.USE_INTENSITY:
                pts_input = np.concatenate((chosen_pts, ret_pts_features), axis=1)  # (N, C)
            else:
                pts_input = chosen_pts
            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = chosen_pts
            sample_info['pts_features'] = ret_pts_features
            return sample_info

        # select subset of bboxes of interest 
        gt_bbox_list_corners = self.filtrate_objects(sample_bboxes)

        # separate points xyz, rgb and classification (not using color data for now)
        xyz = chosen_pts[:,0:3]
        color = chosen_pts[:,3:6]
        classification = chosen_pts[:,6]

        # data augmentation 
        aug_pts_rect = xyz.copy()
        aug_gt_bbox_corners = gt_bbox_list_corners.copy()

        # cfg.AUG_DATA is True
        if cfg.AUG_DATA and self.mode == 'TRAIN':
            aug_pts_rect, aug_gt_bbox_corners, aug_method = self.data_augmentation(aug_pts_rect, aug_gt_bbox_corners)
            sample_info['aug_method'] = aug_method
        # note if we do the augmentation above we only return te augmented scene not the original scene

        # make a 2D np array of features of bboxes
        # output: 2D np array: x,y,z, w,h,l , rx,ry,rz, class_num for each box
        aug_gt_boxes3d = pipe_utils.get_bbox_features_from_corners(aug_gt_bbox_corners)
        

        # the cfg.RPN.USE_INTENSITY in the file is false 
        # prepare input
        if cfg.RPN.USE_INTENSITY:
            pts_input = np.concatenate((aug_pts_rect, ret_pts_features), axis=1)  # (N, C)
        else:
            pts_input = aug_pts_rect

        if cfg.RPN.FIXED:
            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = aug_pts_rect
            sample_info['pts_features'] = ret_pts_features
            sample_info['gt_boxes3d'] = aug_gt_boxes3d
            return sample_info

        # generate training labels
        # pts_input and aug_pts_rect are the same.
        # ret_pts_features is just a zeros column
        rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(aug_pts_rect, aug_gt_boxes3d, gt_bbox_list_corners) 
        sample_info['pts_input'] = pts_input
        sample_info['pts_rect'] = aug_pts_rect
        sample_info['pts_features'] = ret_pts_features
        sample_info['rpn_cls_label'] = rpn_cls_label
        sample_info['rpn_reg_label'] = rpn_reg_label
        sample_info['gt_boxes3d'] = aug_gt_boxes3d


        return sample_info


    @staticmethod
    def generate_rpn_training_labels(pts_rect, gt_boxes3d, gt_corners):
        '''
        Inputs: 
            pts_rect : points in the scene
            gt_boxes3d: the boxes in the form of x,y,z, w,h,l , rx,ry,rz, class_num
            gt_bbox_list_corners: 8 box corners corresponding to the aug_gt_boxes3d list 
            but in a dictionary with keys class numbers (string)

        Returns:
            cls_label: sets the class of each point inside the original bboxes to 1 other points 0.

            reg_label: for regression target(reg_label) we use the same w,h,l , rx,ry,rz as the bbox for each point belonging to that bbox. 
            but the x,y,z of the center of the box are transformed into dx, dy, dz (box center subtracted from point's coordinates)
            order of each row : dx, dy, dz, w,h,l , rx,ry,rz
        '''
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        reg_label = np.zeros((pts_rect.shape[0], 9), dtype=np.float32)  # dx, dy, dz, ry, h, w, l

        # turn gt_corners into a simple 2D np array
        bbox_corners = []
        for _, value in gt_corners.items():
            for corners in value:
                bbox_corners.append(corners)
        gt_corners = np.asarray(bbox_corners)


        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]

            # determin which points inside the box (it's a mask) (Trues are inside)
            fg_pt_flag = kitti_utils.in_hull(pts_rect, box_corners)
            fg_pts_rect = pts_rect[fg_pt_flag]
            cls_label[fg_pt_flag] = 1

            # pixel offset of object center
            center3d = gt_boxes3d[k][0:3].copy()  # (x, y, z)

            # center all points inside box around the box center
            # I have no idea why this is written as center-points instead of points-center which would be more natural
            reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_rect  

            # size and angle encoding
            reg_label[fg_pt_flag, 3] = gt_boxes3d[k][3]  # w     
            reg_label[fg_pt_flag, 4] = gt_boxes3d[k][4]  # h
            reg_label[fg_pt_flag, 5] = gt_boxes3d[k][5]  # l
            reg_label[fg_pt_flag, 6] = gt_boxes3d[k][6]  # rx
            reg_label[fg_pt_flag, 7] = gt_boxes3d[k][7]  # ry
            reg_label[fg_pt_flag, 8] = gt_boxes3d[k][8]  # rz
 

        # check all the returns are non zero (at-least 1 nonzero element)
        #assert cls_label.any(), 'cls_label is all 0s !'
        #assert reg_label.any(), 'reg_label is all 0s !'

        return cls_label, reg_label

   

    def data_augmentation(self, aug_pts_rect, bbox_corners_dict, mustaug=False):
        """
        :param aug_pts_rect: (N, 3)
        :param aug_gt_boxes3d: (N, 7)
        :param gt_alpha: (N)
        :return: the new_aug_bboxes returned is the corners in the form of a dictionary of classes.
        """
        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)
        if mustaug is True:
            aug_enable[0] = -1
            aug_enable[1] = -1
        aug_method = []

        # cfg.AUG_METHOD_PROB = [1.0, 1.0, 0.5]
        if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            # get random rotation angles 10 degrees max 
            angles = np.random.uniform(-np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE, size=3)
            # get the rotation matrix
            R = pipe_utils.euler_Angles_To_Rotation_Matrix(angles)
            # each dimension is in a column so we multiply R from right
            aug_pts_rect = aug_pts_rect @ R
            aug_method.append(['rotation', angles])

        if 'scaling' in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
            scale = np.random.uniform(0.5, 1.5)
            aug_pts_rect = aug_pts_rect * scale
            aug_method.append(['scaling', scale])

        if 'flip' in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
            # flipping other axis can be added later
            # flip horizontal
            aug_pts_rect[:,0] = -aug_pts_rect[:,0]
            aug_method.append('flip')

        # now go throw each bbox and apply the same transforms to their corners
        # in the same order
        new_aug_bboxes = {}
        for class_num, class_bboxes in bbox_corners_dict.items():
            cur_class_bboxes = []
            for bbox in class_bboxes:
                # apply each of the transforms used
                if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
                    bbox = bbox @ R
                if 'scaling' in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
                    bbox = bbox * scale
                if 'flip' in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
                    bbox[:,0] = -bbox[:,0]

                cur_class_bboxes.append(bbox)
            new_aug_bboxes[class_num] = np.asarray(cur_class_bboxes)
                

        return aug_pts_rect, new_aug_bboxes, aug_method


    '''def get_rcnn_sample_info(self, roi_info):
        sample_id, gt_box3d = roi_info['sample_id'], roi_info['gt_box3d']
        rpn_xyz, rpn_features, rpn_intensity, seg_mask = self.rpn_feature_list[sample_id]

        # augmentation original roi by adding noise
        roi_box3d = self.aug_roi_by_noise(roi_info)

        # point cloud pooling based on roi_box3d
        pooled_boxes3d = kitti_utils.enlarge_box3d(roi_box3d.reshape(1, 7), cfg.RCNN.POOL_EXTRA_WIDTH)

        boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(torch.from_numpy(rpn_xyz),
                                                                 torch.from_numpy(pooled_boxes3d))
        pt_mask_flag = (boxes_pts_mask_list[0].numpy() == 1)
        cur_pts = rpn_xyz[pt_mask_flag].astype(np.float32)

        # data augmentation
        aug_pts = cur_pts.copy()
        aug_gt_box3d = gt_box3d.copy().astype(np.float32)
        aug_roi_box3d = roi_box3d.copy()
        if cfg.AUG_DATA and self.mode == 'TRAIN':
            # calculate alpha by ry
            temp_boxes3d = np.concatenate([aug_roi_box3d.reshape(1, 7), aug_gt_box3d.reshape(1, 7)], axis=0)
            temp_x, temp_z, temp_ry = temp_boxes3d[:, 0], temp_boxes3d[:, 2], temp_boxes3d[:, 6]
            temp_beta = np.arctan2(temp_z, temp_x).astype(np.float64)
            temp_alpha = -np.sign(temp_beta) * np.pi / 2 + temp_beta + temp_ry

            # data augmentation
            aug_pts, aug_boxes3d, aug_method = self.data_augmentation(aug_pts, temp_boxes3d, temp_alpha, mustaug=True, stage=2)
            aug_roi_box3d, aug_gt_box3d = aug_boxes3d[0], aug_boxes3d[1]
            aug_gt_box3d = aug_gt_box3d.astype(gt_box3d.dtype)

        # Pool input points
        valid_mask = 1  # whether the input is valid

        if aug_pts.shape[0] == 0:
            pts_features = np.zeros((1, 128), dtype=np.float32)
            input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
            pts_input = np.zeros((1, input_channel), dtype=np.float32)
            valid_mask = 0
        else:
            pts_features = rpn_features[pt_mask_flag].astype(np.float32)
            pts_intensity = rpn_intensity[pt_mask_flag].astype(np.float32)

            pts_input_list = [aug_pts, pts_intensity.reshape(-1, 1)]
            if cfg.RCNN.USE_INTENSITY:
                pts_input_list = [aug_pts, pts_intensity.reshape(-1, 1)]
            else:
                pts_input_list = [aug_pts]

            if cfg.RCNN.USE_MASK:
                if cfg.RCNN.MASK_TYPE == 'seg':
                    pts_mask = seg_mask[pt_mask_flag].astype(np.float32)
                elif cfg.RCNN.MASK_TYPE == 'roi':
                    pts_mask = roipool3d_utils.pts_in_boxes3d_cpu(torch.from_numpy(aug_pts),
                                                                  torch.from_numpy(aug_roi_box3d.reshape(1, 7)))
                    pts_mask = (pts_mask[0].numpy() == 1).astype(np.float32)
                else:
                    raise NotImplementedError

                pts_input_list.append(pts_mask.reshape(-1, 1))

            if cfg.RCNN.USE_DEPTH:
                pts_depth = np.linalg.norm(aug_pts, axis=1, ord=2)
                pts_depth_norm = (pts_depth / 70.0) - 0.5
                pts_input_list.append(pts_depth_norm.reshape(-1, 1))

            pts_input = np.concatenate(pts_input_list, axis=1)  # (N, C)

        aug_gt_corners = kitti_utils.boxes3d_to_corners3d(aug_gt_box3d.reshape(-1, 7))
        aug_roi_corners = kitti_utils.boxes3d_to_corners3d(aug_roi_box3d.reshape(-1, 7))
        iou3d = kitti_utils.get_iou3d(aug_roi_corners, aug_gt_corners)
        cur_iou = iou3d[0][0]

        # regression valid mask
        reg_valid_mask = 1 if cur_iou >= cfg.RCNN.REG_FG_THRESH and valid_mask == 1 else 0

        # classification label
        cls_label = 1 if cur_iou > cfg.RCNN.CLS_FG_THRESH else 0
        if cfg.RCNN.CLS_BG_THRESH < cur_iou < cfg.RCNN.CLS_FG_THRESH or valid_mask == 0:
            cls_label = -1

        # canonical transform and sampling
        pts_input_ct, gt_box3d_ct = self.canonical_transform(pts_input, aug_roi_box3d, aug_gt_box3d)
        pts_input_ct, pts_features = self.rcnn_input_sample(pts_input_ct, pts_features)

        sample_info = {'sample_id': sample_id,
                       'pts_input': pts_input_ct,
                       'pts_features': pts_features,
                       'cls_label': cls_label,
                       'reg_valid_mask': reg_valid_mask,
                       'gt_boxes3d_ct': gt_box3d_ct,
                       'roi_boxes3d': aug_roi_box3d,
                       'roi_size': aug_roi_box3d[3:6],
                       'gt_boxes3d': aug_gt_box3d}

        return sample_info

    @staticmethod
    def canonical_transform(pts_input, roi_box3d, gt_box3d):
        roi_ry = roi_box3d[6] % (2 * np.pi)  # 0 ~ 2pi
        roi_center = roi_box3d[0:3]
        # shift to center
        pts_input[:, [0, 1, 2]] = pts_input[:, [0, 1, 2]] - roi_center
        gt_box3d_ct = np.copy(gt_box3d)
        gt_box3d_ct[0:3] = gt_box3d_ct[0:3] - roi_center
        # rotate to the direction of head
        gt_box3d_ct = kitti_utils.rotate_pc_along_y(gt_box3d_ct.reshape(1, 7), roi_ry).reshape(7)
        gt_box3d_ct[6] = gt_box3d_ct[6] - roi_ry
        pts_input = kitti_utils.rotate_pc_along_y(pts_input, roi_ry)

        return pts_input, gt_box3d_ct

    @staticmethod
    def canonical_transform_batch(pts_input, roi_boxes3d, gt_boxes3d):
        """
        :param pts_input: (N, npoints, 3 + C)
        :param roi_boxes3d: (N, 7)
        :param gt_boxes3d: (N, 7)
        :return:
        """
        roi_ry = roi_boxes3d[:, 6] % (2 * np.pi)  # 0 ~ 2pi
        roi_center = roi_boxes3d[:, 0:3]
        # shift to center
        pts_input[:, :, [0, 1, 2]] = pts_input[:, :, [0, 1, 2]] - roi_center.reshape(-1, 1, 3)
        gt_boxes3d_ct = np.copy(gt_boxes3d)
        gt_boxes3d_ct[:, 0:3] = gt_boxes3d_ct[:, 0:3] - roi_center
        # rotate to the direction of head
        gt_boxes3d_ct = kitti_utils.rotate_pc_along_y_torch(torch.from_numpy(gt_boxes3d_ct.reshape(-1, 1, 7)),
                                                            torch.from_numpy(roi_ry)).numpy().reshape(-1, 7)
        gt_boxes3d_ct[:, 6] = gt_boxes3d_ct[:, 6] - roi_ry
        pts_input = kitti_utils.rotate_pc_along_y_torch(torch.from_numpy(pts_input), torch.from_numpy(roi_ry)).numpy()

        return pts_input, gt_boxes3d_ct

    @staticmethod
    def rcnn_input_sample(pts_input, pts_features):
        choice = np.random.choice(pts_input.shape[0], cfg.RCNN.NUM_POINTS, replace=True)

        if pts_input.shape[0] < cfg.RCNN.NUM_POINTS:
            choice[:pts_input.shape[0]] = np.arange(pts_input.shape[0])
            np.random.shuffle(choice)
        pts_input = pts_input[choice]
        pts_features = pts_features[choice]

        return pts_input, pts_features

    def aug_roi_by_noise(self, roi_info):
        """
        add noise to original roi to get aug_box3d
        :param roi_info:
        :return:
        """
        roi_box3d, gt_box3d = roi_info['roi_box3d'], roi_info['gt_box3d']
        original_iou = roi_info['iou3d']
        temp_iou = cnt = 0
        pos_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_box3d.reshape(-1, 7))
        aug_box3d = roi_box3d
        while temp_iou < pos_thresh and cnt < 10:
            if roi_info['type'] == 'gt':
                aug_box3d = self.random_aug_box3d(roi_box3d)  # GT, must random
            else:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d  # p=0.2 to keep the original roi box
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
            aug_corners = kitti_utils.boxes3d_to_corners3d(aug_box3d.reshape(-1, 7))
            iou3d = kitti_utils.get_iou3d(aug_corners, gt_corners)
            temp_iou = iou3d[0][0]
            cnt += 1
            if original_iou < pos_thresh:  # original bg, break
                break
        return aug_box3d

    @staticmethod
    def random_aug_box3d(box3d):
        """
        :param box3d: (7) [x, y, z, h, w, l, ry]
        random shift, scale, orientation
        """
        if cfg.RCNN.REG_AUG_METHOD == 'single':
            pos_shift = (np.random.rand(3) - 0.5)  # [-0.5 ~ 0.5]
            hwl_scale = (np.random.rand(3) - 0.5) / (0.5 / 0.15) + 1.0  #
            angle_rot = (np.random.rand(1) - 0.5) / (0.5 / (np.pi / 12))  # [-pi/12 ~ pi/12]

            aug_box3d = np.concatenate([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale,
                                        box3d[6:7] + angle_rot])
            return aug_box3d
        elif cfg.RCNN.REG_AUG_METHOD == 'multiple':
            # pos_range, hwl_range, angle_range, mean_iou
            range_config = [[0.2, 0.1, np.pi / 12, 0.7],
                            [0.3, 0.15, np.pi / 12, 0.6],
                            [0.5, 0.15, np.pi / 9, 0.5],
                            [0.8, 0.15, np.pi / 6, 0.3],
                            [1.0, 0.15, np.pi / 3, 0.2]]
            idx = np.random.randint(len(range_config))

            pos_shift = ((np.random.rand(3) - 0.5) / 0.5) * range_config[idx][0]
            hwl_scale = ((np.random.rand(3) - 0.5) / 0.5) * range_config[idx][1] + 1.0
            angle_rot = ((np.random.rand(1) - 0.5) / 0.5) * range_config[idx][2]

            aug_box3d = np.concatenate([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot])
            return aug_box3d
        elif cfg.RCNN.REG_AUG_METHOD == 'normal':
            x_shift = np.random.normal(loc=0, scale=0.3)
            y_shift = np.random.normal(loc=0, scale=0.2)
            z_shift = np.random.normal(loc=0, scale=0.3)
            h_shift = np.random.normal(loc=0, scale=0.25)
            w_shift = np.random.normal(loc=0, scale=0.15)
            l_shift = np.random.normal(loc=0, scale=0.5)
            ry_shift = ((np.random.rand() - 0.5) / 0.5) * np.pi / 12

            aug_box3d = np.array([box3d[0] + x_shift, box3d[1] + y_shift, box3d[2] + z_shift, box3d[3] + h_shift,
                                  box3d[4] + w_shift, box3d[5] + l_shift, box3d[6] + ry_shift])
            return aug_box3d
        else:
            raise NotImplementedError

    def get_proposal_from_file(self, index):
        sample_id = int(self.image_idx_list[index])
        proposal_file = os.path.join(self.rcnn_eval_roi_dir, '%06d.txt' % sample_id)
        roi_obj_list = kitti_utils.get_objects_from_label(proposal_file)

        rpn_xyz, rpn_features, rpn_intensity, seg_mask = self.get_rpn_features(self.rcnn_eval_feature_dir, sample_id)
        pts_rect, pts_rpn_features, pts_intensity = rpn_xyz, rpn_features, rpn_intensity

        roi_box3d_list, roi_scores = [], []
        for obj in roi_obj_list:
            box3d = np.array([obj.pos[0], obj.pos[1], obj.pos[2], obj.h, obj.w, obj.l, obj.ry], dtype=np.float32)
            roi_box3d_list.append(box3d.reshape(1, 7))
            roi_scores.append(obj.score)

        roi_boxes3d = np.concatenate(roi_box3d_list, axis=0)  # (N, 7)
        roi_scores = np.array(roi_scores, dtype=np.float32)  # (N)

        if cfg.RCNN.ROI_SAMPLE_JIT:
            sample_dict = {'sample_id': sample_id,
                           'rpn_xyz': rpn_xyz,
                           'rpn_features': rpn_features,
                           'seg_mask': seg_mask,
                           'roi_boxes3d': roi_boxes3d,
                           'roi_scores': roi_scores,
                           'pts_depth': np.linalg.norm(rpn_xyz, ord=2, axis=1)}

            if self.mode != 'TEST':
                gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
                gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

                roi_corners = kitti_utils.boxes3d_to_corners3d(roi_boxes3d)
                gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d)
                iou3d = kitti_utils.get_iou3d(roi_corners, gt_corners)
                if gt_boxes3d.shape[0] > 0:
                    gt_iou = iou3d.max(axis=1)
                else:
                    gt_iou = np.zeros(roi_boxes3d.shape[0]).astype(np.float32)

                sample_dict['gt_boxes3d'] = gt_boxes3d
                sample_dict['gt_iou'] = gt_iou
            return sample_dict

        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [pts_intensity.reshape(-1, 1), seg_mask.reshape(-1, 1)]
        else:
            pts_extra_input_list = [seg_mask.reshape(-1, 1)]

        if cfg.RCNN.USE_DEPTH:
            cur_depth = np.linalg.norm(pts_rect, axis=1, ord=2)
            cur_depth_norm = (cur_depth / 70.0) - 0.5
            pts_extra_input_list.append(cur_depth_norm.reshape(-1, 1))

        pts_extra_input = np.concatenate(pts_extra_input_list, axis=1)
        pts_input, pts_features = roipool3d_utils.roipool3d_cpu(roi_boxes3d, pts_rect, pts_rpn_features,
                                                                pts_extra_input, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                                sampled_pt_num=cfg.RCNN.NUM_POINTS)

        sample_dict = {'sample_id': sample_id,
                       'pts_input': pts_input,
                       'pts_features': pts_features,
                       'roi_boxes3d': roi_boxes3d,
                       'roi_scores': roi_scores,
                       'roi_size': roi_boxes3d[:, 3:6]}

        if self.mode == 'TEST':
            return sample_dict

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        gt_boxes3d = np.zeros((gt_obj_list.__len__(), 7), dtype=np.float32)

        for k, obj in enumerate(gt_obj_list):
            gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
                = obj.pos, obj.h, obj.w, obj.l, obj.ry

        if gt_boxes3d.__len__() == 0:
            gt_iou = np.zeros((roi_boxes3d.shape[0]), dtype=np.float32)
        else:
            roi_corners = kitti_utils.boxes3d_to_corners3d(roi_boxes3d)
            gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d)
            iou3d = kitti_utils.get_iou3d(roi_corners, gt_corners)
            gt_iou = iou3d.max(axis=1)
        sample_dict['gt_boxes3d'] = gt_boxes3d
        sample_dict['gt_iou'] = gt_iou

        return sample_dict

    def get_rcnn_training_sample_batch(self, index):
        sample_id = int(self.sample_id_list[index])
        rpn_xyz, rpn_features, rpn_intensity, seg_mask = \
            self.get_rpn_features(self.rcnn_training_feature_dir, sample_id)

        # load rois and gt_boxes3d for this sample
        roi_file = os.path.join(self.rcnn_training_roi_dir, '%06d.txt' % sample_id)
        roi_obj_list = kitti_utils.get_objects_from_label(roi_file)
        roi_boxes3d = kitti_utils.objs_to_boxes3d(roi_obj_list)
        # roi_scores = kitti_utils.objs_to_scores(roi_obj_list)

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        # calculate original iou
        iou3d = kitti_utils.get_iou3d(kitti_utils.boxes3d_to_corners3d(roi_boxes3d),
                                      kitti_utils.boxes3d_to_corners3d(gt_boxes3d))
        max_overlaps, gt_assignment = iou3d.max(axis=1), iou3d.argmax(axis=1)
        max_iou_of_gt, roi_assignment = iou3d.max(axis=0), iou3d.argmax(axis=0)
        roi_assignment = roi_assignment[max_iou_of_gt > 0].reshape(-1)

        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_IMAGE))
        fg_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
        fg_inds = np.nonzero(max_overlaps >= fg_thresh)[0]
        fg_inds = np.concatenate((fg_inds, roi_assignment), axis=0)  # consider the roi which has max_overlaps with gt as fg

        easy_bg_inds = np.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH_LO))[0]
        hard_bg_inds = np.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH) &
                                  (max_overlaps >= cfg.RCNN.CLS_BG_THRESH_LO))[0]

        fg_num_rois = fg_inds.size
        bg_num_rois = hard_bg_inds.size + easy_bg_inds.size

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
            rand_num = np.random.permutation(fg_num_rois)
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE  - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(cfg.RCNN.ROI_PER_IMAGE ) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num]
            fg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
            bg_rois_per_this_image = 0
        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)
            fg_rois_per_this_image = 0
        else:
            import pdb
            pdb.set_trace()
            raise NotImplementedError

        # augment the rois by noise
        roi_list, roi_iou_list, roi_gt_list = [], [], []
        if fg_rois_per_this_image > 0:
            fg_rois_src = roi_boxes3d[fg_inds].copy()
            gt_of_fg_rois = gt_boxes3d[gt_assignment[fg_inds]]
            fg_rois, fg_iou3d = self.aug_roi_by_noise_batch(fg_rois_src, gt_of_fg_rois, aug_times=10)
            roi_list.append(fg_rois)
            roi_iou_list.append(fg_iou3d)
            roi_gt_list.append(gt_of_fg_rois)

        if bg_rois_per_this_image > 0:
            bg_rois_src = roi_boxes3d[bg_inds].copy()
            gt_of_bg_rois = gt_boxes3d[gt_assignment[bg_inds]]
            bg_rois, bg_iou3d = self.aug_roi_by_noise_batch(bg_rois_src, gt_of_bg_rois, aug_times=1)
            roi_list.append(bg_rois)
            roi_iou_list.append(bg_iou3d)
            roi_gt_list.append(gt_of_bg_rois)

        rois = np.concatenate(roi_list, axis=0)
        iou_of_rois = np.concatenate(roi_iou_list, axis=0)
        gt_of_rois = np.concatenate(roi_gt_list, axis=0)

        # collect extra features for point cloud pooling
        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [rpn_intensity.reshape(-1, 1), seg_mask.reshape(-1, 1)]
        else:
            pts_extra_input_list = [seg_mask.reshape(-1, 1)]

        if cfg.RCNN.USE_DEPTH:
            pts_depth = (np.linalg.norm(rpn_xyz, ord=2, axis=1) / 70.0) - 0.5
            pts_extra_input_list.append(pts_depth.reshape(-1, 1))
        pts_extra_input = np.concatenate(pts_extra_input_list, axis=1)

        pts_input, pts_features, pts_empty_flag = roipool3d_utils.roipool3d_cpu(rois, rpn_xyz, rpn_features,
                                                                                pts_extra_input,
                                                                                cfg.RCNN.POOL_EXTRA_WIDTH,
                                                                                sampled_pt_num=cfg.RCNN.NUM_POINTS,
                                                                                canonical_transform=False)

        # data augmentation
        if cfg.AUG_DATA and self.mode == 'TRAIN':
            for k in range(rois.__len__()):
                aug_pts = pts_input[k, :, 0:3].copy()
                aug_gt_box3d = gt_of_rois[k].copy()
                aug_roi_box3d = rois[k].copy()

                # calculate alpha by ry
                temp_boxes3d = np.concatenate([aug_roi_box3d.reshape(1, 7), aug_gt_box3d.reshape(1, 7)], axis=0)
                temp_x, temp_z, temp_ry = temp_boxes3d[:, 0], temp_boxes3d[:, 2], temp_boxes3d[:, 6]
                temp_beta = np.arctan2(temp_z, temp_x).astype(np.float64)
                temp_alpha = -np.sign(temp_beta) * np.pi / 2 + temp_beta + temp_ry

                # data augmentation
                aug_pts, aug_boxes3d, aug_method = self.data_augmentation(aug_pts, temp_boxes3d, temp_alpha,
                                                                          mustaug=True, stage=2)

                # assign to original data
                pts_input[k, :, 0:3] = aug_pts
                rois[k] = aug_boxes3d[0]
                gt_of_rois[k] = aug_boxes3d[1]

        valid_mask = (pts_empty_flag == 0).astype(np.int32)

        # regression valid mask
        reg_valid_mask = (iou_of_rois > cfg.RCNN.REG_FG_THRESH).astype(np.int32) & valid_mask

        # classification label
        cls_label = (iou_of_rois > cfg.RCNN.CLS_FG_THRESH).astype(np.int32)
        invalid_mask = (iou_of_rois > cfg.RCNN.CLS_BG_THRESH) & (iou_of_rois < cfg.RCNN.CLS_FG_THRESH)
        cls_label[invalid_mask] = -1
        cls_label[valid_mask == 0] = -1

        # canonical transform and sampling
        pts_input_ct, gt_boxes3d_ct = self.canonical_transform_batch(pts_input, rois, gt_of_rois)

        sample_info = {'sample_id': sample_id,
                       'pts_input': pts_input_ct,
                       'pts_features': pts_features,
                       'cls_label': cls_label,
                       'reg_valid_mask': reg_valid_mask,
                       'gt_boxes3d_ct': gt_boxes3d_ct,
                       'roi_boxes3d': rois,
                       'roi_size': rois[:, 3:6],
                       'gt_boxes3d': gt_of_rois}

        return sample_info

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds, bg_rois_per_this_image):
        if hard_bg_inds.size > 0 and easy_bg_inds.size > 0:
            hard_bg_rois_num = int(bg_rois_per_this_image * cfg.RCNN.HARD_BG_RATIO)
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_num = np.floor(np.random.rand(hard_bg_rois_num) * hard_bg_inds.size).astype(np.int32)
            hard_bg_inds = hard_bg_inds[rand_num]
            # sampling easy bg
            rand_num = np.floor(np.random.rand(easy_bg_rois_num) * easy_bg_inds.size).astype(np.int32)
            easy_bg_inds = easy_bg_inds[rand_num]

            bg_inds = np.concatenate([hard_bg_inds, easy_bg_inds], axis=0)
        elif hard_bg_inds.size > 0 and easy_bg_inds.size == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_num = np.floor(np.random.rand(hard_bg_rois_num) * hard_bg_inds.size).astype(np.int32)
            bg_inds = hard_bg_inds[rand_num]
        elif hard_bg_inds.size == 0 and easy_bg_inds.size > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_num = np.floor(np.random.rand(easy_bg_rois_num) * easy_bg_inds.size).astype(np.int32)
            bg_inds = easy_bg_inds[rand_num]
        else:
            raise NotImplementedError

        return bg_inds

    def aug_roi_by_noise_batch(self, roi_boxes3d, gt_boxes3d, aug_times=10):
        """
        :param roi_boxes3d: (N, 7)
        :param gt_boxes3d: (N, 7)
        :return:
        """
        iou_of_rois = np.zeros(roi_boxes3d.shape[0], dtype=np.float32)
        for k in range(roi_boxes3d.__len__()):
            temp_iou = cnt = 0
            roi_box3d = roi_boxes3d[k]
            gt_box3d = gt_boxes3d[k]
            pos_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
            gt_corners = kitti_utils.boxes3d_to_corners3d(gt_box3d.reshape(1, 7))
            aug_box3d = roi_box3d
            while temp_iou < pos_thresh and cnt < aug_times:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d  # p=0.2 to keep the original roi box
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
                aug_corners = kitti_utils.boxes3d_to_corners3d(aug_box3d.reshape(1, 7))
                iou3d = kitti_utils.get_iou3d(aug_corners, gt_corners)
                temp_iou = iou3d[0][0]
                cnt += 1
            roi_boxes3d[k] = aug_box3d
            iou_of_rois[k] = temp_iou
        return roi_boxes3d, iou_of_rois

    def get_rcnn_sample_jit(self, index):
        sample_id = int(self.sample_id_list[index])
        rpn_xyz, rpn_features, rpn_intensity, seg_mask = \
            self.get_rpn_features(self.rcnn_training_feature_dir, sample_id)

        # load rois and gt_boxes3d for this sample
        roi_file = os.path.join(self.rcnn_training_roi_dir, '%06d.txt' % sample_id)
        roi_obj_list = kitti_utils.get_objects_from_label(roi_file)
        roi_boxes3d = kitti_utils.objs_to_boxes3d(roi_obj_list)
        # roi_scores = kitti_utils.objs_to_scores(roi_obj_list)

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        sample_info = {'sample_id': sample_id,
                       'rpn_xyz': rpn_xyz,
                       'rpn_features': rpn_features,
                       'rpn_intensity': rpn_intensity,
                       'seg_mask': seg_mask,
                       'roi_boxes3d': roi_boxes3d,
                       'gt_boxes3d': gt_boxes3d,
                       'pts_depth': np.linalg.norm(rpn_xyz, ord=2, axis=1)}

        return sample_info
'''
    def collate_batch(self, batch):
        '''
        we need to implement this since at-least for RPN we have a different number of bboxes in each scene
        but since we have an equal number of points the rest of the features of the scene are simpler to batch

        also the code implements the loss function itself so we organize the batch the way we want 
        '''
        if self.mode != 'TRAIN' and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            assert batch.__len__() == 1
            return batch[0]

        batch_size = batch.__len__()
        ans_dict = {}

        # iterate over the keys of the dict we return for 1 sample 
        # then inside this loop iterate over each sample of the batch
        for key in batch[0].keys():
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                # max number of ground truth bboxes of the scenes in this batch
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, batch[k][key].__len__())

                # put all the scenes' bboxes(bbox features) in a single array 
                # (the number of bboxes in each scene can be different)
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 10), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes3d[i, :batch[i][key].__len__(), :] = batch[i][key]
                ans_dict[key] = batch_gt_boxes3d
                continue
            
            # for RPN everything else is an np.ndarray
            # here we concatenate the point features with a new axis
            if isinstance(batch[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = batch[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)

            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict

# for debug
'''
if __name__ == '__main__':
    current_path = '/cvlabdata2/home/farivar/my_network/pipe_detection/tools'
    DATA_PATH = os.path.join(current_path, '../data')
    dataset = pipes_rcnn_dataset(root_dir=DATA_PATH, 
                                npoints=cfg.RPN.NUM_POINTS, 
                                split='train', mode='TRAIN',
                                classes=cfg.CLASSES)

    num = dataset.__len__()

    for i in range(num):
        dataset.__getitem__(i)
        print('one done')
'''
    

