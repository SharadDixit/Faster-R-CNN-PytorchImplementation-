import torch.nn as nn
from RPNDir.AnchorTargetLayer import AnchorTargetLayer
from RPNDir.ProposalLayer import ProposalLayer


class RPN(nn.Module):

    def __init__(self, din):
        super(RPN, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchorScales = [8, 16, 32]
        self.anchorRatios = [0.5, 1, 2]
        self.featStride = [16, ]

        # define the convrelu layers processing input feature map
        self.RPNConv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.ncScoreOut = len(self.anchorScales) * len(self.anchorRatios) * 2  # 2(bg/fg) * 9 (anchors)
        self.RPNClsScore = nn.Conv2d(512, self.ncScoreOut, 1, 1, 0)

        # define anchor box offset prediction layer
        self.ncBboxOut = len(self.anchorScales) * len(self.anchorRatios) * 4  # 4(coords) * 9 (anchors)
        self.RPNBboxPred = nn.Conv2d(512, self.ncBboxOut, 1, 1, 0)

        # define proposal layer
        self.RPNProposal = ProposalLayer(self.featStride, self.anchorScales, self.anchorRatios)

        # define anchor target layer
        self.RPNAnchorTarget = AnchorTargetLayer(self.featStride, self.anchorScales, self.anchorRatios)

        self.rpnLossCls = 0
        self.rpnLossBox = 0