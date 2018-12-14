from __future__ import print_function

import numpy as np


def createAnchors(base_size=16, ratios=[0.5, 1, 2],
                    scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    baseAnchor = np.array([1, 1, base_size, base_size]) - 1
    ratioAnchors = ratioEnum(baseAnchor, ratios)
    anchors = np.vstack([scaleEnum(ratioAnchors[i, :], scales)
                         for i in range(ratioAnchors.shape[0])])
    return anchors


def getWHCentre(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    xCtr = anchor[0] + 0.5 * (w - 1)
    yCtr = anchor[1] + 0.5 * (h - 1)
    return w, h, xCtr, yCtr


def mkanchors(ws, hs, xCtr, yCtr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((xCtr - 0.5 * (ws - 1),
                         yCtr - 0.5 * (hs - 1),
                         xCtr + 0.5 * (ws - 1),
                         yCtr + 0.5 * (hs - 1)))
    return anchors


def ratioEnum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, xCtr, yCtr = getWHCentre(anchor)
    size = w * h
    sizeRatios = size / ratios
    ws = np.round(np.sqrt(sizeRatios))
    hs = np.round(ws * ratios)
    anchors = mkanchors(ws, hs, xCtr, yCtr)
    return anchors


def scaleEnum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = getWHCentre(anchor)
    ws = w * scales
    hs = h * scales
    anchors = mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


if __name__ == '__main__':
    import time

    t = time.time()
    a = createAnchors()
    print(time.time() - t)
    print(a)

