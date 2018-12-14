import torch


def smoothL1Loss(bboxPred, roisTarget, roisInsideWs, roisOutsideWs, sigma=1.0, dim=[1]):
    sigma2 = sigma ** 2
    boxDifference = bboxPred - roisTarget
    inBoxDiff = roisInsideWs * boxDifference

    absInBoxDiff = torch.abs(inBoxDiff)
    smoothL1Sign = (absInBoxDiff < 1. / sigma2).detach().float()
    inLossBox = torch.pow(inBoxDiff, 2) * (sigma2 / 2.) * smoothL1Sign \
                + (absInBoxDiff - (0.5 / sigma2)) * (1. - smoothL1Sign)

    outLossBox = roisOutsideWs * inLossBox
    lossBox = outLossBox

    for i in sorted(dim, reverse=True):
        lossBox = lossBox.sum(i)
    lossBox = lossBox.mean()
    return lossBox