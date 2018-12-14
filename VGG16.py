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

        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)
        print("check")
        self.RCNN_top = vgg.classifier

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.nClasses)

        self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.nClasses)


    # Flattening the features from 7X7 to 1X49

    def headToTail(self, pool5):

        pool5Flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5Flat)

        return fc7
