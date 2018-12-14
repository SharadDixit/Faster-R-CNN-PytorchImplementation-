from torch.nn.modules.module import Module

class RoIPooling(Module):
    def __init__(self, pooledHeight, pooledWidth, spatialScale):
        super(RoIPooling, self).__init__()

        self.pooled_width = int(pooledWidth)
        self.pooled_height = int(pooledHeight)
        self.spatial_scale = float(spatialScale)