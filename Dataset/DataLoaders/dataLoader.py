import torch
import numpy as np
import os
import heapq
import networkx as nx


datadir = "./Dataset/Data"


def healthCheckOnRoiSignal(roiSignal):
    """
        roiSignal : (N, T)
    """

    # remove subjects with dead rois
    # if(np.sum(np.sum(np.abs(roiSignal), axis=1) == 0) > 0):
    #     return False
    if(roiSignal.shape[1] <= 128):
        return False
    return True

def abide1Loader(atlas, targetTask, sort = True):

    """
        x : (#subjects, N)
    """

    dataset = torch.load(datadir + "/dataset_abide_{}.save".format(atlas))
    sorted_indices = ECSort()

    x = []
    y = []
    subjectIds = []

    for data in dataset:
        if(targetTask == "disease"):
            label = int(data["pheno"]["disease"]) - 1 # 0 for autism 1 for control

        if(healthCheckOnRoiSignal(data["roiTimeseries"].T)):
            if (sort == True):
                timeseries = data["roiTimeseries"].T
                ROI = timeseries[sorted_indices,:]
            else:
                ROI = data["roiTimeseries"].T
            x.append(ROI)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))

    return x, y, subjectIds


def ADHDLoader(atlas, targetTask, sort = True):
    """
        x : (#subjects, N)
    """

    dataset = torch.load(datadir + "/dataset_ADHD_{}.save".format(atlas))
    sorted_indices = ECSort()

    x = []
    y = []
    subjectIds = []

    for data in dataset:

        if (targetTask == "disease"):
            if(data["pheno"]["disease"] > 0):
                label = 1
            else:
                label = 0


        if (healthCheckOnRoiSignal(data["roiTimeseries"].T)):
            if (sort == True):
                timeseries = data["roiTimeseries"].T
                ROI = timeseries[sorted_indices,:]
            else:
                ROI = data["roiTimeseries"].T
            x.append(ROI)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))

    return x, y, subjectIds
