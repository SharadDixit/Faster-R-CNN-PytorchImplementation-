from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d


class RoIAlignAvg(Module):
    def __init__(self, alignedHeight, alignedWidth, spatialScale):
        super(RoIAlignAvg, self).__init__()

        self.alignedWidth = int(alignedWidth)
        self.alignedHeight = int(alignedHeight)
        self.spatialScale = float(spatialScale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.alignedHeight + 1, self.alignedWidth + 1,
                             self.spatialScale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)
