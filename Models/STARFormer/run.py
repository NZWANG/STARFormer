from tqdm import tqdm
import torch
import numpy as np
import random
import os
import sys


if(not "utils" in os.getcwd()):
    sys.path.append("../../../")

from utils import Option, calculateMetric

from Models.STARFormer.model import Model
from Dataset.dataset import getDataset, getDataset_test


def train(model, dataset, dataset_test, fold, nOfEpochs):
    dataLoader = dataset.getFold(fold, train=True)
    dataLoader_test = dataset_test.getFold(fold, train=False)

    accuracy = 0.
    for epoch in range(nOfEpochs):

        preds = []
        probs = []
        groundTruths = []
        losses = []

        for i, data in enumerate(tqdm(dataLoader,file=sys.stdout, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):

            xTrain = data["timeseries"] # (batchSize, N, dynamicLength)
            yTrain = data["label"] # (batchSize, )

            #xTrain and yTrain are still on "cpu" at this point
            train_loss, train_preds, train_probs, yTrain = model.step(xTrain, yTrain, train=True)


            preds.append(train_preds)
            probs.append(train_probs)
            groundTruths.append(yTrain)
            losses.append(train_loss)
        print("lr:", model.optimizer.param_groups[0]["lr"])
        preds = torch.cat(preds, dim=0).numpy()
        probs = torch.cat(probs, dim=0).numpy()
        groundTruths = torch.cat(groundTruths, dim=0).numpy()
        losses = torch.tensor(losses).numpy()

        print("Epoch {} loss : {}".format(epoch, losses.mean()))

        metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
        print("Train metrics : {}".format(metrics))

        preds_ = []
        probs_ = []
        groundTruths_ = []
        losses_ = []


        for i, data in enumerate(tqdm(dataLoader_test,file=sys.stdout,ncols=60, desc=f'Testing fold:{fold}')):
            xTest = data["timeseries"]
            yTest = data["label"]

            #xTrain and yTrain are still on "cpu" at this point
            test_loss, test_preds, test_probs, yTest = model.step(xTest, yTest, train=False)


            preds_.append(test_preds)
            probs_.append(test_probs)
            groundTruths_.append(yTest)
            losses_.append(test_loss)

        preds_ = torch.cat(preds_, dim=0).numpy()
        probs_ = torch.cat(probs_, dim=0).numpy()
        groundTruths_ = torch.cat(groundTruths_, dim=0).numpy()
        loss_ = torch.tensor(losses_).numpy().mean()

        print("Test loss : {}".format(loss_))
        metrics_ = calculateMetric({"predictions": preds_, "probs": probs_, "labels": groundTruths_})
        print("Test metrics : {}".format(metrics_))

        if(accuracy <= metrics_["accuracy"]):
            loss_test = loss_
            preds_test = preds_
            probs_test = probs_
            groundTruths_test = groundTruths_
            accuracy = metrics_["accuracy"]
            metrics_test = metrics_
            print("Best test metrics : {}\n".format(metrics_test))
    print("Best test metrics : {}".format(metrics_test))
    return preds, probs, groundTruths, losses, preds_test, probs_test, groundTruths_test, loss_test

def run_STARFormer(hyperParams, datasetDetails, device):

    foldCount = datasetDetails.foldCount
    datasetSeed = datasetDetails.datasetSeed
    nOfEpochs = datasetDetails.nOfEpochs
    dynamicLength = datasetDetails.dynamicLength

    dataset = getDataset(datasetDetails)
    dataset_test = getDataset_test(datasetDetails)

    details = Option({
        "device" : device,
        "nOfTrains" : dataset.get_nOfTrains_perFold(),
        "nOfClasses" : datasetDetails.nOfClasses,
        "batchSize" : datasetDetails.batchSize,
        "nOfEpochs" : nOfEpochs,
        "dynamicLength" : dynamicLength
    })

    results = []

    for fold in range(foldCount):

        model = Model(hyperParams, details)

        train_preds, train_probs, train_groundTruths, train_loss, test_preds, test_probs, test_groundTruths, test_loss = train(model, dataset, dataset_test, fold, nOfEpochs)

        result = {
            "train" : {
                "labels" : train_groundTruths,
                "predictions" : train_preds,
                "probs" : train_probs,
                "loss" : train_loss
            },
            "test" : {
                "labels" : test_groundTruths,
                "predictions" : test_preds,
                "probs" : test_probs,
                "loss" : test_loss
            }
        }
        results.append(result)

    return results
