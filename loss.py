import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math
from torch.nn.modules import loss


def class_select(logits, target):
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)


class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2, alpha=0.25, aggregate='mean'):
        super(FocalLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate
        self.alpha = alpha
        # self.alpha = Variable(torch.ones(num_classes)*alpha)
        self.gamma = gamma
        self.num_classes = num_classes
        print('Initializing FocalLoss for training: alpha={}, gamma={}'.format(self.alpha, self.gamma))

    def forward(self, input, target, weights=None):
        assert input.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1

        logpt = F.log_softmax(input, dim=1)
        logpt_gt = logpt.gather(1,target.unsqueeze(1))
        logpt_gt = logpt_gt.view(-1)
        pt_gt = logpt_gt.exp()
        assert logpt_gt.size() == pt_gt.size()
        
        loss = -self.alpha*(torch.pow((1-pt_gt), self.gamma))*logpt_gt
        
        if self.aggregate == 'sum':
            return loss.sum()
        elif self.aggregate == 'mean':
            return loss.mean()
        elif self.aggregate is None:
            return loss

class InstanceCrossEntropyLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean', weighted=0):
        super(InstanceCrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate
        self.weighted = weighted
        print('Initializing InstanceCrossEntropyLoss for training: with weights{}'.format(self.weighted))
        if self.weighted == 1:
            print('Weighted loss is used...')

    def forward(self, logits, target, weights=None):
        assert logits.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1
        softmax_result = F.log_softmax(logits, dim=1)
        loss = class_select(-softmax_result, target)

        if self.weighted == 1 or self.weighted == 2:
            assert list(loss.size()) == list(weights.size())
            loss = weights * loss

        if self.aggregate == 'sum':
            return loss.sum()
        elif self.aggregate == 'mean':
            return loss.mean()
        elif self.aggregate is None:
            return loss


class SmoothlabelCrossEntropyLoss(nn.Module):
    def __init__(self, beta=1.0, aggregate='mean', weighted=0):
        super(SmoothlabelCrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate
        self.weighted = weighted
        self.beta = beta
        print('Initializing SmoothlabelCrossEntropyLoss for training: beta={}, weights={}'.format(self.beta, self.weighted))
        if self.weighted == 1:
            print('Weighted loss is used...')
            
    def forward(self, input, target, weights=None):
        assert input.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1

        logpt = F.log_softmax(input, dim=1)
        logpt_gt = logpt.gather(1,target.unsqueeze(1))
        logpt_gt = logpt_gt.view(-1)
        logpt_pred,_ = torch.max(logpt,1)
        logpt_pred = logpt_pred.view(-1)
        assert logpt_gt.size() == logpt_pred.size()
        loss = - logpt_gt - self.beta* logpt_pred
        
        if self.weighted == 1 or self.weighted == 2:
            assert list(loss.size()) == list(weights.size())
            loss = loss * weights
        if self.aggregate == 'sum':
            return loss.sum()
        elif self.aggregate == 'mean':
            return loss.mean()
        elif self.aggregate is None:
            return loss

class SmoothlabelClassCrossEntropyLoss(nn.Module):
    def __init__(self, beta=0.0, aggregate='mean', weighted=0):
        super(SmoothlabelClassCrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate
        self.weighted = weighted
        self.beta = beta
        print('Initializing SmoothlabelClassCrossEntropyLoss for training: beta={}, weights={}'.format(self.beta, self.weighted))
        if self.weighted == 1:
            print('Weighted loss is used...')
            
    def forward(self, input, target, weights=None):
        assert input.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1

        logpt = F.log_softmax(input, dim=1)
        logpt_gt = logpt.gather(1,target.unsqueeze(1))
        logpt_gt = logpt_gt.view(-1)
        logpt_pred,_ = torch.max(logpt,1)
        logpt_pred = logpt_pred.view(-1)
        assert logpt_gt.size() == logpt_pred.size()
        loss = - (1-self.beta)*logpt_gt - self.beta* logpt_pred
        
        if self.weighted == 1:
            assert list(loss.size()) == list(weights.size())
            loss = loss * weights.exp()
        if self.aggregate == 'sum':
            return loss.sum()
        elif self.aggregate == 'mean':
            return loss.mean()
        elif self.aggregate is None:
            return loss
            
class LabelRefineLoss(nn.Module):
    def __init__(self, lambda1=0.0, aggregate='mean'):
        super(LabelRefineLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate
        self.lambda1 = lambda1
        print('Initializing LabelRefineLoss for training: lambda1={}'.format(self.lambda1))
            
    def forward(self, input, target, lambda1):
        assert input.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1

        logpt = F.log_softmax(input, dim=1)
        logpt_gt = logpt.gather(1,target.unsqueeze(1))
        logpt_gt = logpt_gt.view(-1)
        logpt_pred,_ = torch.max(logpt,1)
        logpt_pred = logpt_pred.view(-1)
        assert logpt_gt.size() == logpt_pred.size()
        loss = - (1-lambda1)*logpt_gt - lambda1* logpt_pred
        
        if self.aggregate == 'sum':
            return loss.sum()
        elif self.aggregate == 'mean':
            return loss.mean()
        elif self.aggregate is None:
            return loss
            
class InstanceWeightLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean', weighted=0):
        super(InstanceWeightLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate
        self.weighted = weighted
        print('Initializing Instance Weight for training: with weights{}'.format(self.weighted))
        if self.weighted == 1:
            print('Weighted loss is used...')

    def forward(self, logits, target, weights=None):
        assert logits.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1
        softmax_result = F.log_softmax(logits, dim=1)
        loss = class_select(-softmax_result, target)

        if self.weighted == 1 or self.weighted == 2:
            assert list(loss.size()) == list(weights.size())
            # pdb.set_trace()
            loss = weights * loss

        if self.aggregate == 'sum':
            return loss.sum()
        elif self.aggregate == 'mean':
            return loss.mean()
        elif self.aggregate is None:
            return loss

class CoRefineLoss(loss._Loss):

    def __init__(self, lambda1=0.0, aggregate='mean'):
        super(CoRefineLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate
        self.lambda1 = lambda1

        """The KL-Divergence loss for the model and refined labels output.
        output must be a pair of (model_output, refined_labels), both NxC tensors.
        The rows of refined_labels must all add up to one (probability scores);
        however, model_output must be the pre-softmax output of the network."""

    def forward(self, output1, output2, target, lambdaKL = 0):

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        if output2.requires_grad:
            raise ValueError("Refined labels should not require gradients.")

        output1_log_prob = F.log_softmax(output1, dim=1)
        output2_prob = F.softmax(output2, dim=1)

        _, pred_label = output2_prob.max(1)

        # Loss is normal cross entropy loss
        base_loss = F.cross_entropy(output1, pred_label)


        # Loss is -dot(model_output_log_prob, refined_labels). Prepare tensors
        # for batch matrix multiplicatio

        model_output1_log_prob = output1_log_prob.unsqueeze(2)
        model_output2_prob = output2_prob.unsqueeze(1)

        # Compute the loss, and average/sum for the batch.
        kl_loss = -torch.bmm(model_output2_prob, model_output1_log_prob)
        if self.aggregate == 'mean':
            loss_co = base_loss.mean() + lambdaKL * kl_loss.mean()
        else:
            loss_co = base_loss.sum() + lambdaKL * kl_loss.sum()
        return loss_co

class CoRefineLossPLus(loss._Loss):

        def __init__(self, lambda1=0.0, aggregate='mean'):
            super(CoRefineLossPLus, self).__init__()
            assert aggregate in ['sum', 'mean', None]
            self.aggregate = aggregate
            self.lambda1 = lambda1

            """The KL-Divergence loss for the model and refined labels output.
            output must be a pair of (model_output, refined_labels), both NxC tensors.
            The rows of refined_labels must all add up to one (probability scores);
            however, model_output must be the pre-softmax output of the network."""

        def forward(self, output1, output2, target, lambdaKL=0):

            # Target is ignored at training time. Loss is defined as KL divergence
            # between the model output and the refined labels.
            if output2.requires_grad:
                raise ValueError("Refined labels should not require gradients.")

            output1_log_prob = F.log_softmax(output1, dim=1)
            output2_prob = F.softmax(output2, dim=1)

            _, pred_label2 = output2_prob.max(1)
            _, pred_label1 = output1_log_prob.max(1)

            # compute the mask
            mask = pred_label2.eq(pred_label1)
            # Loss is normal cross entropy loss
            base_loss = F.cross_entropy(output1, pred_label2)
            base_loss = base_loss * mask.float()


            # Loss is -dot(model_output_log_prob, refined_labels). Prepare tensors
            # for batch matrix multiplicatio

            model_output1_log_prob = output1_log_prob.unsqueeze(2)
            model_output2_prob = output2_prob.unsqueeze(1)


            # Compute the loss, and average/sum for the batch.
            kl_loss = -torch.bmm(model_output2_prob, model_output1_log_prob)
            if self.aggregate == 'mean':
                loss_co = base_loss.mean() + lambdaKL * kl_loss.mean()
            else:
                loss_co = base_loss.sum() + lambdaKL * kl_loss.sum()
            return loss_co