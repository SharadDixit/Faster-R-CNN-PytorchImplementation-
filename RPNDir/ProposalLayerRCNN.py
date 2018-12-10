import torch.nn as nn
import torch


class ProposalTargetLayer(nn.Module):

    def __init__(self, nclasses):
        super(ProposalTargetLayer, self).__init__()
        self.numClasses = nclasses
        "To normalize the target using Means and Standard deviation"

        self.BBOXNORMALIZEMEANS = torch.FloatTensor((0.0, 0.0, 0.0, 0.0))
        self.BBOXNORMALIZESTDS = torch.FloatTensor((0.1, 0.1, 0.2, 0.2))
        self.BBOXINSIDEWEIGHTS = torch.FloatTensor((1.0, 1.0, 1.0, 1.0))
