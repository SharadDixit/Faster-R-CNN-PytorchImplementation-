import torch.nn as nn
from RPNDir.RPN import RPN
from RPNDir.ProposalLayerRCNN import ProposalTargetLayer
from ROI.RoIPooling import RoIPooling
from ROI.RoIAlignAvg import RoIAlignAvg
from ROI.RoICrop import RoICrop


class FasterRCNN(nn.Module):
    "Initialize classes and init network"
    def __init__(self, classes):
        super(FasterRCNN, self).__init__()
        self.classes = classes
        self.nClasses = len(classes)
        # loss
        self.RCNNLossCls = 0
        self.RCNNLossBbox = 0

        self.RCNNRpn = RPN(self.doutBaseModel)
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

        self.normalInit(self.RCNNRpn.RPNConv, 0, 0.01, False)
        self.normalInit(self.RCNNRpn.RPNClsScore, 0, 0.01, False)
        self.normalInit(self.RCNNRpn.RPNBboxPred, 0, 0.01, False)
        self.normalInit(self.RCNNClsScore, 0, 0.01, False)
        self.normalInit(self.RCNNBboxPred, 0, 0.001, False)

    def createArchitecture(self):
        self.initModules()
        self.initWeights()