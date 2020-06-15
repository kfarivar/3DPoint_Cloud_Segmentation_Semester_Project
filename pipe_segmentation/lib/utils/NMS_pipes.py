import torch
import numpy as np

from lib.config import cfg

def pipes_nms(bbox_features, cls_scores, thresh):
    '''perform non maximum supression to select promissing bboxes for 1 scene
    bbox_features: (N,9)
    cls_scores: the raw output of classification (no sigmoid performed) (N,1)
    thresh: the radius from the current box centers that we use to exclude in nms
    '''
    # sort the indexes of cls_score 
    _, sorted_idxs = torch.sort(cls_scores, dim=0, descending=True)

    # sort by classification score
    proposals_ordered = bbox_features[sorted_idxs]

    # choose the top 9000 points from the sorted list
    top_proposals = proposals_ordered[:cfg['TEST'].RPN_PRE_NMS_TOP_N].cpu().numpy()

    # perform the NMS
    # save the bboxes we want to return here
    # we start with the highest scoring bbox
    selected_bboxes = [top_proposals[0]] 
    for box in top_proposals:
        promissing = True
        for selected_box in selected_bboxes:
            # calculate the distance between the center of two bbboxes
            # if they are too close the current box is not promissing
            distance_centers = np.linalg.norm(box[0:3]-selected_box[0:3])
            if distance_centers < thresh:
                promissing = False
                break
        if promissing:
            selected_bboxes.append(box)
        # if we have all the boxes we asked for finish it (100 boxes max)
        if len(selected_bboxes) >= cfg['TEST'].RPN_POST_NMS_TOP_N:
            break
    
    # note the returned bboxes might be less than 100
    selected_bboxes = np.asarray(selected_bboxes)  
    return selected_bboxes
