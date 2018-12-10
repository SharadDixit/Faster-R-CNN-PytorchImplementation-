from FasterRCNN import FasterRCNN
import torchvision.models as models
import torch.nn as nn


class VGG16(FasterRCNN):
    def __init__(self, classes):
        self.doutBaseModel = 512

        FasterRCNN.__init__(self, classes)

    def initModules(self):
        vgg = models.vgg16()

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        # not using the last maxpool layer

        self.RCNNBase = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.RCNNBase[layer].parameters(): p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        self.RCNNTop = vgg.classifier

        # not using the last maxpool layer
        self.RCNNClsScore = nn.Linear(4096, self.nClasses)

        self.RCNNBboxPred = nn.Linear(4096, 4 * self.nClasses)

