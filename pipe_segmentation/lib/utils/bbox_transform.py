import torch
import numpy as np
from scipy.spatial.distance import cdist

from lib.utils.pipes_bbox_utils import get_bbox_corners_from_features


def decode_bbox_target(roi_box3d, pred_reg):
    """
    :param roi_box3d: (N, 3) don't know why this is N by 7 in original code but in our application this is the xyz of the backbone and it is (N,3)
    :param pred_reg: (N, 9) dx, dy, dz, w,h,l , rx,ry,rz
    :return:
    """
    # we do a normal regression so this is much simpler

    # note previously for the bbox-center label of each point we had (center - point-coord) so now we 
    # add the location of the point back to the prediction we have made for the center of the bbox 
    pos_xyz = pred_reg[:, 0:3] + roi_box3d[:, 0:3]
    
    whl = pred_reg[:, 3:6]

    # apply the atan function to the angles we have predicted 
    # rpn_reg has shape (B,N,9) the last three features are angles
    angles = torch.atan(pred_reg[:,-3:])

    # concatenate
    ret_box3d = torch.cat((pos_xyz.view(-1, 3), whl.view(-1, 3), angles.view(-1, 3)), dim=1)

    return ret_box3d


def nms_calculate_corners_error (pred_boxes, gt_boxes):
    ''' this function calculates the avg corners error for 1 scene for the predicted bboxes that have been filtered 
    using non maximum supression. every bbox is paired with the ground truth bbox that has the closest center.
    pred_boxes: predicted bboxes (K,9)
    gt_boxes: ground truth bboxes (M,10) last column is class number (not important for now) 
    '''
    gt_boxes = gt_boxes.cpu().numpy()[:,:-1] # exclude the class num

    # calculate distance matrix
    distance_matrix = cdist(pred_boxes[0:3], gt_boxes[0:3])

    # in each row (for each predicted bbox) find the closest gt_box
    closest_box_idxs = np.argmin(distance_matrix, axis=1)
    # get the corresponding closest bbox
    closest_bbox = gt_boxes[closest_box_idxs]

    # calculate the corners error
    total_error=0
    for pred_box, gt_box in zip(pred_boxes,closest_bbox):
        # get the corners for each box
        predicted_corners = get_bbox_corners_from_features(pred_box)
        gt_corners = get_bbox_corners_from_features(gt_box) 

        # the corresponding corners match each other since we use the same code to make them
        # we should also make sure bboxes are uniquely defined by their 9 features.
        errors = predicted_corners - gt_corners
        
        # calculate the norm of the error for each corner seperately then take average
        cur_error = np.mean(np.sqrt(np.sum(errors**2, axis=1)))

        total_error += cur_error
    
    # normalize by the total number of predicted boxes
    return total_error/ pred_boxes.shape[0]


def calculate_bbox_corners_error(predicted_bbox_features, label_bbox_features, rpn_cls_label):
    ''' calculate the bbox corners and then the error for 1 scene
    only points that are labeled inside a bbox are considered

    predicted_bbox_features: (N,9)

    returns: the mean of the norm 2 error of all the corners in this scene
    '''

    # calculate error only for the points that have classification label == 1
    mask = rpn_cls_label > 0
    predicted_bbox_features = predicted_bbox_features[mask].cpu().numpy()
    label_bbox_features = label_bbox_features[mask].cpu().numpy()
    
    # it would be better if this was vectorized
    total_error = 0
    num_points = len(predicted_bbox_features)
    for idx in range(num_points):

        # get the corners for each point
        predicted_corners = get_bbox_corners_from_features( predicted_bbox_features[idx] )
        label_corners = get_bbox_corners_from_features( label_bbox_features[idx] )

        # the corresponding corners match each other since we use the same code to make them
        # we should also make sure bboxes are uniqely defined by their 9 features.
        errors = predicted_corners - label_corners
        
        # calculate the norm of the error for each point seperately then take average
        cur_error = np.mean(np.sqrt(np.sum(errors**2, axis=1)))

        total_error += cur_error
    
    return  total_error / num_points




'''
def rotate_pc_along_y_torch(pc, rot_angle):
    """
    :param pc: (N, 3 + C)
    :param rot_angle: (N)
    :return:
    """
    cosa = torch.cos(rot_angle).view(-1, 1)
    sina = torch.sin(rot_angle).view(-1, 1)

    raw_1 = torch.cat([cosa, -sina], dim=1)
    raw_2 = torch.cat([sina, cosa], dim=1)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)), dim=1)  # (N, 2, 2)

    pc_temp = pc[:, [0, 2]].unsqueeze(dim=1)  # (N, 1, 2)

    pc[:, [0, 2]] = torch.matmul(pc_temp, R.permute(0, 2, 1)).squeeze(dim=1)
    return pc'''