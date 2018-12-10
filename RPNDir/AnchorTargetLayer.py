import torch.nn as nn
import torch
import numpy as np
import RPNDir.Anchors as Anchors

class AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, featStride, scales, ratios):
        super(AnchorTargetLayer, self).__init__()

        self._feat_stride = featStride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(Anchors.createAnchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0