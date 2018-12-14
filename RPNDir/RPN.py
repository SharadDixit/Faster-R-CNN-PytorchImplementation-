import torch.nn as nn
from RPNDir.AnchorTargetLayer import AnchorTargetLayer
from RPNDir.ProposalLayer import ProposalLayer
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from ROI.LossFunctions import smoothL1Loss

class RPN(nn.Module):

    def __init__(self, din):
        super(RPN, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchorScales = [8, 16, 32]
        self.anchorRatios = [0.5, 1, 2]
        self.featStride = [16, ]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.ncScoreOut = len(self.anchorScales) * len(self.anchorRatios) * 2  # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.ncScoreOut, 1, 1, 0)

        # define anchor box offset prediction layer
        self.ncBboxOut = len(self.anchorScales) * len(self.anchorRatios) * 4  # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.ncBboxOut, 1, 1, 0)

        # define proposal layer
        self.RPNProposal = ProposalLayer(self.featStride, self.anchorScales, self.anchorRatios)

        # define anchor target layer
        self.RPNAnchorTarget = AnchorTargetLayer(self.featStride, self.anchorScales, self.anchorRatios)

        self.rpnLossCls = 0
        self.rpnLossBox = 0

    def reshape(self,x, d):
        inputShape = x.size()
        x = x.view(
            inputShape[0],
            int(d),
            int(float(inputShape[1] * inputShape[2]) / float(d)),
            inputShape[3]
        )
        return x

    def forward(self, basefeatureMap, imageInfo, groundTruthBoxes, numBoxes):
        batch_size = basefeatureMap.size(0)

        # return feature map after convrelu layer
        rpnConv1 = F.relu(self.RPN_Conv(basefeatureMap), inplace=True)
        # get rpn classification score
        rpnClsScore = self.RPN_cls_score(rpnConv1)

        rpnClsScoreReshape = self.reshape(rpnClsScore, 2)
        rpnClsProbReshape = F.softmax(rpnClsScoreReshape, 1)
        rpnClsProb = self.reshape(rpnClsProbReshape, self.ncScoreOut)

        # get rpn offsets to the anchor boxes
        rpnBboxPred = self.RPN_bbox_pred(rpnConv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPNProposal((rpnClsProb.data, rpnBboxPred.data,
                                  imageInfo, cfg_key))

        self.rpnLossCls = 0
        self.rpnLossBox = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert groundTruthBoxes is not None

            rpnData = self.RPNAnchorTarget((rpnClsScore.data, groundTruthBoxes, imageInfo, numBoxes))

            # compute classification loss
            rpnClsScore = rpnClsScoreReshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpnLabel = rpnData[0].view(batch_size, -1)

            rpnKeep = Variable(rpnLabel.view(-1).ne(-1).nonzero().view(-1))
            rpnClsScore = torch.index_select(rpnClsScore.view(-1, 2), 0, rpnKeep)
            rpnLabel = torch.index_select(rpnLabel.view(-1), 0, rpnKeep.data)
            rpnLabel = Variable(rpnLabel.long())
            self.rpnLossCls = F.cross_entropy(rpnClsScore, rpnLabel)
            fg_cnt = torch.sum(rpnLabel.data.ne(0))  # xxxxx calculated but not used

            rpnBboxTargets, rpnBboxInsideWeights, rpnBboxOutsideWeights = rpnData[1:]

            # compute bbox regression loss
            rpnBboxInsideWeights = Variable(rpnBboxInsideWeights)
            rpnBboxOutsideWeights = Variable(rpnBboxOutsideWeights)
            rpnBboxTargets = Variable(rpnBboxTargets)

            self.rpnLossBox = smoothL1Loss(rpnBboxPred, rpnBboxTargets, rpnBboxInsideWeights,
                                           rpnBboxOutsideWeights, sigma=3, dim=[1, 2, 3])

        return rois, self.rpnLossCls, self.rpnLossBox

