import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.rpn.proposal_layer import ProposalLayer
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
import lib.utils.loss_utils as loss_utils
from lib.config import cfg
import importlib


class RPN(nn.Module):
    def __init__(self, use_xyz=True, mode='TRAIN'):
        super().__init__()
        self.training_mode = (mode == 'TRAIN')

        MODEL = importlib.import_module(cfg.RPN.BACKBONE)
        self.backbone_net = MODEL.get_model(input_channels=int(cfg.RPN.USE_INTENSITY), use_xyz=use_xyz)

        # here Conv1d is almost the same as torch Conv1d
        # for torch Conv1d see https://pytorch.org/docs/stable/nn.html#conv1d
        # here we use the Conv1d so we can do two levels of batch calculation 
        # the first level is at the level of the scenes and the second is at the level of th points
        # The input to both heads is a (B, C, N) shaped tensor.  
        # C is number of channels (i.e. the number of features each point has) (it is apparently 128)
        # N is the number of points in one scene 
        # this way we regress the output values of all the points using a single run of a Conv1d layer 
        # Notice the output has the form: classification head  (B,1,N) , regression head (B,9,N)
        # since the kernel_size is 1 the output is a linear combination of channels just like a simple linear regression plus a bias
        # in the case of the regression head, each of the 9 outputs has its own set of weights and biases.  
        # notice the output is the result of the regression/classification for all the points not just a single one.

        # classification branch
        cls_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]  # = 128
        for k in range(0, cfg.RPN.CLS_FC.__len__()):
            # input is 128 output is also 128
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.CLS_FC[k], bn=cfg.RPN.USE_BN)) # bn is batch normalization
            pre_channel = cfg.RPN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None)) # sigmoid is applied in the loss function not here
        # this ends up being: 
        # 1st layer 128 inputs to 128 outputs
        # 2nd layer 128 to 1 
        if cfg.RPN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_cls_layer = nn.Sequential(*cls_layers)
        # it adds a dropout layer with ratio 0.5
        

        # regression branch
        # we will do a normal regression for all the 9 parameters (x,y,z, w,h,l , rx,ry,rz) of our bboxes 
        reg_channel = 9  

        reg_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]  # = 128
        for k in range(0, cfg.RPN.REG_FC.__len__()): # cfg.RPN.REG_FC = [128]
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.REG_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.REG_FC[k]

        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None)) 
        
        #if you use binning and classification the activation of this last layer is applied in the loss instead
        # see /lib/utils/loss_utils.py "get_reg_loss" it uses BinaryCrossEntropy which applies a softmax (I need to change this !)

        if cfg.RPN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_reg_layer = nn.Sequential(*reg_layers)
        # this ends up being: 
        # 1st layer 128 inputs to 128 outputs
        # 2nd layer 128 to 9 outputs 
        #  it adds a dropout layer with ratio 0.5

        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            self.rpn_cls_loss_func = loss_utils.DiceLoss(ignore_target=-1)
        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            self.rpn_cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RPN.FOCAL_ALPHA[0],
                                                                               gamma=cfg.RPN.FOCAL_GAMMA)
        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            self.rpn_cls_loss_func = F.binary_cross_entropy
        else:
            raise NotImplementedError
        
        # proposal layer is only used in RCNN and not in RPN
        self.proposal_layer = ProposalLayer(mode=mode)
        self.init_weights()

    def init_weights(self):
        if cfg.RPN.LOSS_CLS in ['SigmoidFocalLoss']:
            pi = 0.01
            nn.init.constant_(self.rpn_cls_layer[2].conv.bias, -np.log((1 - pi) / pi))

        nn.init.normal_(self.rpn_reg_layer[-1].conv.weight, mean=0, std=0.001)

    def forward(self, input_data):
        """
        :param input_data: dict (point_cloud)
        :return:
        """
        pts_input = input_data['pts_input']
        backbone_xyz, backbone_features = self.backbone_net(pts_input)  # (B, N, 3), (B, C, N)

        # backbone_features is in the form (B, C, N) B is batch size(a batch is a set of scenes)
        # C is number of channels (here the number of features each point has apparently 128)
        # N is the number of points in one scene

        # the output of the Conv1d is in the form (B,C,N) so we transpose
        rpn_cls = self.rpn_cls_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, 1)
        rpn_reg = self.rpn_reg_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, C)

        # what is the difference between backbone_xyz and pts_input ?
        ret_dict = {'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg,
                    'backbone_xyz': backbone_xyz, 'backbone_features': backbone_features}

        return ret_dict

