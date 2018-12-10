from torch.nn.modules.module import Module


class RoICrop(Module):
    def __init__(self, layout = 'BHWD'):
        super(RoICrop, self).__init__()
