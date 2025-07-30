from utils import Option

def getHyper_STARFormer():

    hyperDict = {

            "weightDecay" : 0,

            "lr" : 5e-5,
            "minLr" : 1e-6,
            "maxLr" :1e-4,

            "nOfLayers" : 3,
            "dim" : 400,

            "numHeads" : 8,
            "headDim" : 16,

            "windowSize" : 8,
            "shiftCoeff" : 1.0,
            "fringeCoeff" : 2,
            "focalRule" : "expand",

            "mlpRatio" : 1.0,
            "attentionBias" : True,
            "drop" : 0.5,
            "attnDrop" : 0.5,
            "lambdaCons" : 0,
            "pooling" : "gmp",
    }

    return Option(hyperDict)

