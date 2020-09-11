# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 02:48:30 2016

@author: maekawa
"""

import numpy as np


def calcHist(array, hdims, hranges, density=False):
    """
    make histogram of array
    :param array: 1D-array
    :param hdims: int
    :param hranges: list of int
        minvalue, maxvalue
    :return: 1D-array
        histogram
    """
    hist, edges = np.histogram(array, bins=np.linspace(hranges[0], hranges[1], hdims + 1), density=density)
    return hist.astype(dtype=float)


def compareHist(hist1, hist2, method):
    """

    http://opencv.jp/opencv-2svn/cpp/histograms.html
    (Is the formula of CV_COMP_CHISQR in the Document　correct?)
    :param hist1: 1D-array
    :param hist2: 1D-array
    :param method: int
        0: Correlation -[-1, 1]+
        1: Chi-square -[inf, 0]+
        2: Intersect -[0, 1]+
        3: Bhattacharyya -[1, 0]+
    :return: float
        distance between two histogram
    """
    if method % 4 == 0:  # CV_COMP_CORREL
        dist = np.corrcoef(hist1, hist2)[0, 1]

    elif method % 4 == 1:  # CV_COMP_CHISQR
        normalized_hist1 = hist1 / np.sum(hist1)
        normalized_hist2 = hist2 / np.sum(hist2)
        dist = np.sum((np.power(normalized_hist1 - normalized_hist2, 2)[normalized_hist1 != 0]
                       / normalized_hist1[normalized_hist1 != 0]))

    elif method % 4 == 2:  # CV_COMP_INTERSECT
        normalized_hist1 = hist1 / np.sum(hist1)
        normalized_hist2 = hist2 / np.sum(hist2)
        dist = np.sum(np.minimum(normalized_hist1, normalized_hist2))

    else:  # method % 4 == 3:  # CV_COMP_BHATTACHARYYA
        dist = np.sqrt(1. - np.sum(np.sqrt(hist1 * hist2)) / (len(hist1) * np.sqrt(np.mean(hist1) * np.mean(hist2))))

    return dist


def check_cv2():
    import cv2

    sample1 = np.random.normal(0, 1, 10000)
    sample2 = np.random.normal(0.5, 1, 10000)

    hdims = 100
    hranges = [-5.0, 5.0]

    cvhist1 = cv2.calcHist(np.array([sample1.astype(np.float32)]), [0], None, [hdims], hranges)
    cvhist2 = cv2.calcHist(np.array([sample2.astype(np.float32)]), [0], None, [hdims], hranges)

    hist1 = calcHist(sample1, hdims, hranges)
    hist2 = calcHist(sample2, hdims, hranges)

    print('CV_CORREL: ' + str(cv2.compareHist(cvhist1, cvhist2, 0)))
    print('   CORREL: ' + str(compareHist(hist1, hist2, 0)))
    print('CV_CHISQR: ' + str(cv2.compareHist(cvhist1 / np.sum(cvhist1), cvhist2 / np.sum(cvhist2), 1)))
    print('   CHISQR: ' + str(compareHist(hist1, hist2, 1)))
    print('CV_INTERSECT: ' + str(cv2.compareHist(cvhist1 / np.sum(cvhist1), cvhist2 / np.sum(cvhist2), 2)))
    print('   INTERSECT: ' + str(compareHist(hist1, hist2, 2)))
    print('CV_BHATTACHARYYA: ' + str(cv2.compareHist(cvhist1, cvhist2, 3)))
    print('   BHATTACHARYYA: ' + str(compareHist(hist1, hist2, 3)))


if __name__ == '__main__':
    check_cv2()
