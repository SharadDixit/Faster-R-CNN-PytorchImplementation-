import torch
import sys
import os
import numpy as np
from VGG16 import VGG16


if __name__ == '__main__':
    pascalClasses = np.asarray(['__background__',
                                 'aeroplane', 'bicycle', 'bird', 'boat',
                                 'bottle', 'bus', 'car', 'cat', 'chair',
                                 'cow', 'diningtable', 'dog', 'horse',
                                 'motorbike', 'person', 'pottedplant',
                                 'sheep', 'sofa', 'train', 'tvmonitor'])

    checkpoint = torch.load('faster_rcnn_1_6_10021.pth', map_location=(lambda storage, loc: storage))
    print(checkpoint)

    fasterRCNN = VGG16(pascalClasses)

    fasterRCNN.load_state_dict(checkpoint['model'])

    fasterRCNN.createArchitecture()


