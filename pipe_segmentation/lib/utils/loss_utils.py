import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, ignore_target=-1):
        super().__init__()
        self.ignore_target = ignore_target

    def forward(self, input, target):
        """
        :param input: (N), logit
        :param target: (N), {0, 1}
        :return:
        """
        input = torch.sigmoid(input.view(-1))
        target = target.float().view(-1)
        mask = (target != self.ignore_target).float()
        return 1.0 - (torch.min(input, target) * mask).sum() / torch.clamp((torch.max(input, target) * mask).sum(), min=1.0)


class SigmoidFocalClassificationLoss(nn.Module):
    """Sigmoid focal cross entropy loss.
      Focal loss down-weights well classified examples and focusses on the hard
      examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
            gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
            alpha: optional alpha weighting factor to balance positives vs negatives.
            all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self,
                prediction_tensor,
                target_tensor,
                weights):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]
            class_indices: (Optional) A 1-D integer tensor of class indices.
              If provided, computes loss only for the specified class indices.

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor))
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) +
               ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
        return focal_cross_entropy_loss * weights


def _sigmoid_cross_entropy_with_logits(logits, labels):
    # to be compatible with tensorflow, we don't use ignore_idx
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    # transpose_param = [0] + [param[-1]] + param[1:-1]
    # logits = logits.permute(*transpose_param)
    # loss_ftor = nn.NLLLoss(reduce=False)
    # loss = loss_ftor(F.logsigmoid(logits), labels)
    return loss


def get_reg_loss(pred_reg, reg_label, loc_scope, loc_bin_size, num_head_bin, anchor_size,
                 get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25, get_ry_fine=False):

    """
    Bin-based 3D bounding boxes regression loss. See https://arxiv.org/abs/1812.04244 for more details.
    
    :param pred_reg: (N, 9)
    :param reg_label: (N, 9) [dx, dy, dz, w, h, l, rx,ry,rz]
    :param loc_scope: constant
    :param loc_bin_size: constant
    :param num_head_bin: constant
    :param anchor_size: (N, 3) or (3) this is the mean size of the object
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_ry_fine:
    :return:
    """

    reg_loss_dict = {}
    loc_loss = 0

    # using a atan function bring the angles between -pi/2 and pi/2
    # we can't change inplace like below pytorch gives "modified by an inplace operation" error
    #pred_reg[:,-3:] = torch.atan(pred_reg[:,-3:])
    # make a new array new_pred_reg
    new_pred_reg = pred_reg.new_empty(pred_reg.size())
    new_pred_reg[:,:-3] = pred_reg[:,:-3]
    new_pred_reg[:,-3:] = torch.atan(pred_reg[:,-3:])

    # get the loss of each feature 
    all_errors = F.smooth_l1_loss(input=new_pred_reg, target=reg_label, reduction='none')
    all_losses = torch.mean(all_errors, dim=0)

    # save each one separately
    reg_loss_dict['loss_x'], reg_loss_dict['loss_y'], reg_loss_dict['loss_z'] = all_losses[0:3]
    reg_loss_dict['loss_w'],  reg_loss_dict['loss_h'],  reg_loss_dict['loss_l'] = all_losses[3:6]
    reg_loss_dict['loss_rx'],  reg_loss_dict['loss_ry'],  reg_loss_dict['loss_rz'] = all_losses[6:]
    
    # get different losses
    loc_loss = torch.sum(all_losses[0:3])
    size_loss = torch.sum(all_losses[3:6])
    angle_loss = torch.sum(all_losses[6:])

    #get the numeric values
    reg_loss_dict['loss_loc'] = loc_loss.item()
    reg_loss_dict['loss_angle'] = angle_loss.item()
    reg_loss_dict['loss_size'] = size_loss.item()

    return loc_loss, size_loss, angle_loss, reg_loss_dict
