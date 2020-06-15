import torch
import torch.nn as nn
from lib.utils.bbox_transform import decode_bbox_target
from lib.config import cfg
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils


class ProposalLayer(nn.Module):
    def __init__(self, mode='TRAIN'):
        super().__init__()
        self.mode = mode
        self.MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

    def forward(self, rpn_scores, rpn_reg, xyz):
        """
        Note all the inputs are in batch form
        :param rpn_scores: (B, N): This is the regression(raw) output of the classification (no sigmoid applied) B:number of batches, N: number of points
        :param rpn_reg: (B, N, 9) we have 9 box features. (I don't think in the original code the third dimension is 8 it should be much more with all the classification)
        :param xyz: (B, N, 3)
        :return bbox3d: (B, M, 9)
        """
        batch_size = xyz.shape[0]
        proposals = decode_bbox_target(xyz.view(-1, 3), rpn_reg.view(-1, rpn_reg.shape[-1]),
                                       anchor_size=self.MEAN_SIZE,
                                       loc_scope=cfg.RPN.LOC_SCOPE,
                                       loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                                       num_head_bin=cfg.RPN.NUM_HEAD_BIN,
                                       get_xz_fine=cfg.RPN.LOC_XZ_FINE,
                                       get_y_by_bin=False,
                                       get_ry_fine=False)  # (N, 9)

        proposals = proposals.view(batch_size, -1, rpn_reg.shape[-1])

        # for the rpn classification output(raw) the more positive a point is the more likely it is in a box.
        # so we can sort by this value for most likely points in a box.
        scores = rpn_scores
        _, sorted_idxs = torch.sort(scores, dim=1, descending=True)

        batch_size = scores.size(0)
        # new is new_empty in the new pytorch version.
        # Returns a Tensor of given size filled with uninitialized data. 
        # By default, the returned Tensor has the same torch.dtype and torch.device as this tensor.
        ret_bbox3d = scores.new(batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N, rpn_reg.shape[-1]).zero_() # self.mode == 'TEST' , RPN_POST_NMS_TOP_N==100
        ret_scores = scores.new(batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N).zero_()
        for k in range(batch_size):
            # score of all the points in one scene
            scores_single = scores[k]
            # proposed bboxes of 1 scene (each point proposes a bbox)
            proposals_single = proposals[k]
            # sorted indexes of points in this scene accroding to score (first one has the highest score)
            order_single = sorted_idxs[k]

            '''if cfg.TEST.RPN_DISTANCE_BASED_PROPOSE: # this is true
                scores_single, proposals_single = self.distance_based_proposal(scores_single, proposals_single,
                                                                               order_single)
            else:'''
            
            scores_single, proposals_single = self.score_based_proposal(scores_single, proposals_single,
                                                                            order_single)

            proposals_tot = proposals_single.size(0)
            ret_bbox3d[k, :proposals_tot] = proposals_single
            ret_scores[k, :proposals_tot] = scores_single

        return ret_bbox3d, ret_scores

    def distance_based_proposal(self, scores, proposals, order):
        """
         propose rois in two area based on the distance
         The bbox proposals are groupe into two regions far and near.
         out of the total number of bboxes we are going to predict we give 70 precent to the near region,
         and 30 precent to the far. 
         e.g if we are going to select 100 bboxes we select the top 70 boxes from the near (accordign to the scores),
         and top 30from far   
        :param scores: (N)
        :param proposals: (N, 9) the bbox features
        :param order: (N)
        """
        nms_range_list = [0, 40.0, 80.0]
        pre_tot_top_n = cfg[self.mode].RPN_PRE_NMS_TOP_N # 100
        pre_top_n_list = [0, int(pre_tot_top_n * 0.7), pre_tot_top_n - int(pre_tot_top_n * 0.7)] # [0,70,30]
        post_tot_top_n = cfg[self.mode].RPN_POST_NMS_TOP_N
        post_top_n_list = [0, int(post_tot_top_n * 0.7), post_tot_top_n - int(post_tot_top_n * 0.7)]

        scores_single_list, proposals_single_list = [], []

        # sort by score
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]

        # distance is the z of the predicted bbox centers
        dist = proposals_ordered[:, 2] 
        first_mask = (dist > nms_range_list[0]) & (dist <= nms_range_list[1]) # 0 < distance < 40 
        for i in range(1, len(nms_range_list)):
            # get proposal distance mask
            dist_mask = ((dist > nms_range_list[i - 1]) & (dist <= nms_range_list[i]))

            if dist_mask.sum() != 0:
                # this area has points
                # reduce by mask
                cur_scores = scores_ordered[dist_mask]
                cur_proposals = proposals_ordered[dist_mask]

                # fetch pre nms top K
                # but how do we know we will have 70 points ?
                cur_scores = cur_scores[:pre_top_n_list[i]]
                cur_proposals = cur_proposals[:pre_top_n_list[i]]
            else:
                assert i == 2, '%d' % i
                # this area doesn't have any points, so use rois of first area
                cur_scores = scores_ordered[first_mask]
                cur_proposals = proposals_ordered[first_mask]

                # fetch top K of first area
                # choose the 30 bboxes from the first region But the 30 that come after the first top 70.
                cur_scores = cur_scores[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]
                cur_proposals = cur_proposals[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]

            # oriented nms
            boxes_bev = kitti_utils.boxes3d_to_bev_torch(cur_proposals)
            if cfg.RPN.NMS_TYPE == 'rotate':
                keep_idx = iou3d_utils.nms_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
            elif cfg.RPN.NMS_TYPE == 'normal':
                keep_idx = iou3d_utils.nms_normal_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
            else:
                raise NotImplementedError

            # Fetch post nms top k
            keep_idx = keep_idx[:post_top_n_list[i]]

            scores_single_list.append(cur_scores[keep_idx])
            proposals_single_list.append(cur_proposals[keep_idx])

        scores_single = torch.cat(scores_single_list, dim=0)
        proposals_single = torch.cat(proposals_single_list, dim=0)
        return scores_single, proposals_single

    def score_based_proposal(self, scores, proposals, order):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        # sort by score
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]

        # pre nms top K
        cur_scores = scores_ordered[:cfg[self.mode].RPN_PRE_NMS_TOP_N]
        cur_proposals = proposals_ordered[:cfg[self.mode].RPN_PRE_NMS_TOP_N]

        boxes_bev = kitti_utils.boxes3d_to_bev_torch(cur_proposals)
        keep_idx = iou3d_utils.nms_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)

        # Fetch post nms top k
        keep_idx = keep_idx[:cfg[self.mode].RPN_POST_NMS_TOP_N]

        return cur_scores[keep_idx], cur_proposals[keep_idx]



