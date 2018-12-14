import torch.nn as nn
import RPNDir.Anchors as Anchors
import numpy as np
import torch


class ProposalLayer(nn.Module):

    def __init__(self, featStride, scales, ratios):
        super(ProposalLayer, self).__init__()

        self.featStride = featStride
        self.anchors = torch.from_numpy(Anchors.createAnchors(scales=np.array(scales),
            ratios=np.array(ratios))).float()
        self.numAnchors = self.anchors.size(0)
