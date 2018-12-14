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

        self.featStride = featStride
        self.scales = scales
        anchor_scales = scales
        self.anchors = torch.from_numpy(Anchors.createAnchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self.numAnchors = self.anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self.allowedBorder = 0  # default is 0


    def forward(self, input):
        rpnClsScore = input[0]
        groundtruthBoxes = input[1]
        imageInfo = input[2]
        numBoxes = input[3]

        # map of shape (..., H, W)
        height, width = rpnClsScore.size(2), rpnClsScore.size(3)

        batchSize = groundtruthBoxes.size(0)

        featureHeight, featureWidth = rpnClsScore.size(2), rpnClsScore.size(3)
        shift_x = np.arange(0, featureWidth) * self.featStride
        shift_y = np.arange(0, featureHeight) * self.featStride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpnClsScore).float()

        A = self.numAnchors
        K = shifts.size(0)

        self.anchors = self.anchors.type_as(groundtruthBoxes)  # move to specific gpu.
        allAnchors = self.anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        allAnchors = allAnchors.view(K * A, 4)

        total_anchors = int(K * A)

        keep = ((allAnchors[:, 0] >= -self._allowed_border) &
                (allAnchors[:, 1] >= -self._allowed_border) &
                (allAnchors[:, 2] < int(imageInfo[0][1]) + self.allowedBorder) &
                (allAnchors[:, 3] < int(imageInfo[0][0]) + self.allowedBorder))


        inds_inside = torch.nonzero(keep).view(-1)

        # keep only inside anchors
        anchors = allAnchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = groundtruthBoxes.new(batchSize, inds_inside.size(0)).fill_(-1)
        bboxInsideWeights = groundtruthBoxes.new(batchSize, inds_inside.size(0)).zero_()
        bboxOutsideWeights = groundtruthBoxes.new(batchSize, inds_inside.size(0)).zero_()

        overlaps = bbox_overlaps_batch(anchors, groundtruthBoxes) #TODO define function

        maxOverlaps, argmaxOverlaps = torch.max(overlaps, 2)
        gtMaxOverlaps, _ = torch.max(overlaps, 1)

        #if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:  ------ condition always true
        labels[maxOverlaps < 0.3] = 0   #cfg.TRAIN.RPN_NEGATIVE_OVERLAP=.3

        gtMaxOverlaps[gtMaxOverlaps == 0] = 1e-5
        keep = torch.sum(overlaps.eq(gtMaxOverlaps.view(batchSize, 1, -1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            labels[keep > 0] = 1

        # fg label: above threshold IOU
        labels[maxOverlaps >0.7] = 1  #Train.RPN_POSITIVE_OVERLAP=0.7

        #if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        #    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0------------ will always be false

        numFg = int(0.5*256)    #--------(cfg.TRAIN.RPN_FG_FRACTION=0.5 * cfg.TRAIN.RPN_BATCHSIZE=256)

        sumFg = torch.sum((labels == 1).int(), 1)
        sumBg = torch.sum((labels == 0).int(), 1)

        for i in range(batchSize):
            # subsample positive labels if we have too many
            if sumFg[i] > numFg:
                fgInds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                # rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                randNum = torch.from_numpy(np.random.permutation(fgInds.size(0))).type_as(groundtruthBoxes).long()
                disableInds = fgInds[randNum[:fgInds.size(0) - numFg]]
                labels[i][disableInds] = -1

            #           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            numBg = 256 - torch.sum((labels == 1).int(), 1)[i]     #cfg.TRAIN.RPN_BATCHSIZE=256

            # subsample negative labels if we have too many
            if sumBg[i] > numBg:
                bgInds = torch.nonzero(labels[i] == 0).view(-1)
                # rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bgInds.size(0))).type_as(groundtruthBoxes).long()
                disableInds = bgInds[rand_num[:bgInds.size(0) - numBg]]
                labels[i][disableInds] = -1

        offset = torch.arange(0, batchSize) * groundtruthBoxes.size(1)

        argmaxOverlaps = argmaxOverlaps + offset.view(batchSize, 1).type_as(argmaxOverlaps)

        bboxTargets = computeTargetsBatch(anchors,
                                              groundtruthBoxes.view(-1, 5)[argmaxOverlaps.view(-1), :].view(batchSize, -1, 5))

        # use a single value instead of 4 values for easy index.
        bboxInsideWeights[labels == 1] = (1.0, 1.0, 1.0, 1.0)       #cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]=(1.0, 1.0, 1.0, 1.0)

        if -1.0<0:           #cfg.TRAIN.RPN_POSITIVE_WEIGHT=-1.0 =< 0:
            numExamples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / numExamples.item()
            negative_weights = 1.0 / numExamples.item()
        #else:
        #    assert ((-1.0 > 0) &
        #            (-1.0 < 1))

        bboxOutsideWeights[labels == 1] = positive_weights
        bboxOutsideWeights[labels == 0] = negative_weights

        labels = _unmap(labels, total_anchors, inds_inside, batchSize, fill=-1)
        bboxTargets = _unmap(bboxTargets, total_anchors, inds_inside, batchSize, fill=0)
        bboxInsideWeights = _unmap(bboxInsideWeights, total_anchors, inds_inside, batchSize, fill=0)
        bboxOutsideWeights = _unmap(bboxOutsideWeights, total_anchors, inds_inside, batchSize, fill=0)

        outputs = []

        labels = labels.view(batchSize, height, width, A).permute(0, 3, 1, 2).contiguous()
        labels = labels.view(batchSize, 1, A * height, width)
        outputs.append(labels)

        bboxTargets = bboxTargets.view(batchSize, height, width, A * 4).permute(0, 3, 1, 2).contiguous()
        outputs.append(bboxTargets)

        anchors_count = bboxInsideWeights.size(1)
        bboxInsideWeights = bboxInsideWeights.view(batchSize, anchors_count, 1).expand(batchSize, anchors_count,
                                                                                            4)

        bboxInsideWeights = bboxInsideWeights.contiguous().view(batchSize, height, width, 4 * A) \
            .permute(0, 3, 1, 2).contiguous()

        outputs.append(bboxInsideWeights)

        bboxOutsideWeights = bboxOutsideWeights.view(batchSize, anchors_count, 1).expand(batchSize, anchors_count,
                                                                                              4)
        bboxOutsideWeights = bboxOutsideWeights.contiguous().view(batchSize, height, width, 4 * A) \
            .permute(0, 3, 1, 2).contiguous()
        outputs.append(bboxOutsideWeights)

        return outputs


def _unmap(data, count, inds, batchSize, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batchSize, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batchSize, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret

def computeTargetsBatch(ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])       #TODO define function
