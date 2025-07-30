
# parse the arguments

import argparse
import torch
from datetime import datetime


from utils import Option, metricSummer, calculateMetrics, dumpTestResults

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="abide1")
parser.add_argument("-m", "--model", type=str, default="STARFormer")
parser.add_argument("--device", type=int, default='0')
parser.add_argument("--name", type=str, default="noname")

argv = parser.parse_args()

from Dataset.datasetDetails import datasetDetailsDict

# import model runners

from Models.STARFormer.run import run_STARFormer
# import hyper param fetchers
from Models.STARFormer.hyperparams import getHyper_STARFormer

hyperParamDict = {
        "STARFormer" : getHyper_STARFormer,
}

modelDict = {
        "STARFormer" : run_STARFormer,
}


getHyper = hyperParamDict[argv.model]
runModel = modelDict[argv.model]

print("\nTest model is {}".format(argv.model))


datasetName = argv.dataset
datasetDetails = datasetDetailsDict[datasetName]
hyperParams = getHyper()

print("Dataset details : {}".format(datasetDetails))

resultss = []

seeds = []


for i, seed in enumerate(seeds):

    # for reproducability
    torch.manual_seed(seed)

    print("Running the model with seed : {}".format(seed))
    results = runModel(hyperParams, Option({**datasetDetails,"datasetSeed":seed}), device="cuda:{}".format(argv.device))

    resultss.append(results)
    


metricss = calculateMetrics(resultss) 
meanMetrics_seeds, stdMetrics_seeds, meanMetric_all, stdMetric_all = metricSummer(metricss, "test")

# now dump metrics
dumpTestResults(argv.name, hyperParams, argv.model, datasetName, metricss)
print(meanMetrics_seeds)
print("\\n \ n meanMetrics_all : {}".format(meanMetric_all))
print("stdMetric_all : {}".format(stdMetric_all))
