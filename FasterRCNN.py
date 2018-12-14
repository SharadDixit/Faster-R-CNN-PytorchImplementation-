import torch.nn as nn
from RPNDir.RPN import RPN
from RPNDir.ProposalLayerRCNN import ProposalTargetLayer
from ROI.RoIPooling import RoIPooling
from ROI.RoIAlignAvg import RoIAlignAvg
from ROI.RoICrop import RoICrop
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from ROI.LossFunctions import smoothL1Loss

class FasterRCNN(nn.Module):
    "Initialize classes and init network"
    def __init__(self, classes):
        super(FasterRCNN, self).__init__()
        self.classes = classes
        self.nClasses = len(classes)
        # loss
        self.RCNNLossCls = 0
        self.RCNNLossBbox = 0

        self.RCNN_rpn = RPN(self.doutBaseModel)
        self.RCNNProposalTarget = ProposalTargetLayer(self.nClasses)

        self.RCNNRoiPool = RoIPooling(7, 7, 1.0 / 16.0)
        self.RCNNRoiAlign = RoIAlignAvg(7, 7, 1.0 / 16.0)

        self.gridSize = 7 * 2    # cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNNRoiCrop = RoICrop()

    def normalInit(self, m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

    def initWeights(self):

        self.normalInit(self.RCNN_rpn.RPN_Conv, 0, 0.01, False)
        self.normalInit(self.RCNN_rpn.RPN_cls_score, 0, 0.01, False)
        self.normalInit(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, False)
        self.normalInit(self.RCNN_cls_score, 0, 0.01, False)
        self.normalInit(self.RCNN_bbox_pred, 0, 0.001, False)

    def createArchitecture(self):
        self.initModules()
        self.initWeights()

    def smoothL1Loss(self, bboxPred, roisTarget, roisInsideWs, roisOutsideWs, sigma = 1.0 ,dim = [1]):


        sigma2 = sigma ** 2
        boxDifference = bboxPred - roisTarget
        inBoxDiff = roisInsideWs * boxDifference

        absInBoxDiff = torch.abs(inBoxDiff)
        smoothL1Sign = (absInBoxDiff < 1. / sigma2).detach().float()
        inLossBox = torch.pow(inBoxDiff, 2) * (sigma2 / 2.) * smoothL1Sign \
                      + (absInBoxDiff - (0.5 / sigma2)) * (1. - smoothL1Sign)

        outLossBox = roisOutsideWs * inLossBox
        lossBox = outLossBox

        for i in sorted(dim, reverse=True):
            lossBox = lossBox.sum(i)
        lossBox = lossBox.mean()
        return lossBox

    def forward(self, imageData, imageInfo, groundTruthBoxes, numBoxes):
        batchSize = imageData.size(0)

        imageInfo = imageInfo.data
        groundTruthBoxes = groundTruthBoxes.data
        numBoxes = numBoxes.data

        # feed image data to base model to obtain base feature map
        baseFeatureMap = self.RCNN_base(imageData)

        # feed base feature map tp RPN to obtain rois
        roiS, rpnLossCls, rpnLossBbox = self.RCNN_rpn(baseFeatureMap, imageInfo, groundTruthBoxes, numBoxes)

        if self.training:       # xxxxxxxxx    To do
            roiData = self.RCNNProposalTarget(roiS, groundTruthBoxes, numBoxes)
            roiS, roisLabel, roisTarget, roisInsideWs, roisOutsideWs = roiData

            roisLabel = Variable(roisLabel.view(-1).long())
            roisTarget = Variable(roisTarget.view(-1, roisTarget.size(2)))
            roisInsideWs = Variable(roisInsideWs.view(-1, roisInsideWs.size(2)))
            roisOutsideWs = Variable(roisOutsideWs.view(-1, roisOutsideWs.size(2)))
        else:
            roisLabel = None
            roisTarget = None
            roisInsideWs = None
            roisOutsideWs = None
            rpnLossCls = 0
            rpnLossBbox = 0

        roiS = Variable(roiS)
        # do roi pooling based on predicted rois

        pooledFeatures = self.RCNNRoiAlign(baseFeatureMap, roiS.view(-1, 5))    # xxxxxxx function write (RCNN_roi_align)

        # feed pooled features to top model
        pooledFeatures = self.headToTail(pooledFeatures)

        # compute bbox offset
        bboxPred = self.RCNN_bbox_pred(pooledFeatures)

        # xxxxxxxxxx check here
        # # compute bbox offset
        # bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        # if self.training and not self.class_agnostic:
        #     # select the corresponding columns according to roi labels
        #     bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
        #     bbox_pred_select = torch.gather(bbox_pred_view, 1,
        #                                     rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
        #     bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        clsScore = self.RCNN_cls_score(pooledFeatures)
        clsProb = F.softmax(clsScore, 1)

        RCNNLossCls = 0
        RCNNLossBbox = 0

        if self.training:
            # classification loss
            RCNNLossCls = F.cross_entropy(clsScore, roisLabel)

            # bounding box regression L1 loss
            RCNNLossBbox = smoothL1Loss(bboxPred, roisTarget, roisInsideWs, roisOutsideWs)   #xxxxxxxxx check here

        clsProb = clsProb.view(batchSize, roiS.size(1), -1)
        bboxPred = bboxPred.view(batchSize, roiS.size(1), -1)

        print("in fwd ",bboxPred)

        return roiS, clsProb, bboxPred, rpnLossCls, rpnLossBbox, RCNNLossCls, RCNNLossBbox, roisLabel

