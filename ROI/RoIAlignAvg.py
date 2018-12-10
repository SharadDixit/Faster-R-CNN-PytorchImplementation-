from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d


class RoIAlignAvg(Module):
    def __init__(self, alignedHeight, alignedWidth, spatialScale):
        super(RoIAlignAvg, self).__init__()

        self.alignedWidth = int(alignedWidth)
        self.alignedHeight = int(alignedHeight)
        self.spatialScale = float(spatialScale)