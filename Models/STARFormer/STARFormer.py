import torch
from torch import nn

from Models.STARFormer.TransformerBlock import TransformerBlock

class STARFormer(nn.Module):
    def __init__(self, hyperParams, details):

        super().__init__()

        dim = hyperParams.dim
        nOfClasses = details.nOfClasses
        dynamicLength = details.dynamicLength
        self.hyperParams = hyperParams

        self.inputNorm = nn.LayerNorm(dim)

        self.blocks_up_time = []
        self.blocks_down_time = []
        self.block_space = []

        for i, layer in enumerate(range(hyperParams.nOfLayers)):
            windowSize = hyperParams.windowSize * (2 ** layer)
            receptiveSize = windowSize * 2
            shiftSize = int(windowSize * hyperParams.shiftCoeff)
            print("receptiveSize per window for layer {} : {}".format(i, receptiveSize))

            self.blocks_up_time.append(TransformerBlock(
                dim = hyperParams.dim,
                numHeads = hyperParams.numHeads,
                headDim= hyperParams.headDim,
                windowSize = windowSize,
                receptiveSize = receptiveSize,
                shiftSize = shiftSize,
                mlpRatio = hyperParams.mlpRatio,
                attentionBias = hyperParams.attentionBias,
                drop = hyperParams.drop,
                attnDrop = hyperParams.attnDrop
            ))

        self.blocks_up_time = nn.ModuleList(self.blocks_up_time)

        for i, layer in enumerate(reversed(range(hyperParams.nOfLayers))):
            windowSize = hyperParams.windowSize * (2 ** layer)
            receptiveSize = windowSize * 2
            shiftSize = int(windowSize * hyperParams.shiftCoeff)
            print("receptiveSize per window for layer {} : {}".format(i, receptiveSize))

            self.blocks_down_time.append(TransformerBlock(
                dim=hyperParams.dim,
                numHeads=hyperParams.numHeads,
                headDim=hyperParams.headDim,
                windowSize=windowSize,
                receptiveSize=receptiveSize,
                shiftSize=shiftSize,
                mlpRatio=hyperParams.mlpRatio,
                attentionBias=hyperParams.attentionBias,
                drop=hyperParams.drop,
                attnDrop=hyperParams.attnDrop
            ))

        self.blocks_down_time = nn.ModuleList(self.blocks_down_time)

        self.block_space.append(TransformerBlock(
            dim=dynamicLength,
            numHeads=hyperParams.numHeads,
            headDim=hyperParams.headDim,
            windowSize=hyperParams.dim,
            receptiveSize=hyperParams.dim,
            shiftSize=1,
            mlpRatio=hyperParams.mlpRatio,
            attentionBias=hyperParams.attentionBias,
            drop=hyperParams.drop,
            attnDrop=hyperParams.attnDrop
        ))
        self.block_space = nn.ModuleList(self.block_space)

        self.encoder_postNorm = nn.LayerNorm(dim)
        self.classifierHead = nn.Linear(2 * dim, nOfClasses)

    def forward(self, roiSignals):
        roiSignals = roiSignals.permute((0,2,1)) # (batchSize, dynamicLength, N)
        T = roiSignals.shape[1] # dynamicLength

        roiSignals_space = roiSignals.transpose(1, 2) # (batchSize, N, dynamicLength)
        layer = 0
        roi_time = {}
        for block in self.blocks_up_time:
            roiSignals = block(roiSignals) # (batchSize, dynamicLength, N)
            roi_time[layer] = roiSignals
            layer += 1

        for block in self.blocks_down_time:
            layer -= 1
            roiSignals = block(roiSignals)
            roiSignals = roiSignals + roi_time[layer]

        for block in self.block_space:
            roiSignals_space = block(roiSignals_space)

        roiSignals_space = roiSignals_space.transpose(1,2) # (batchSize, dynamicLength, N)
        roiSignals = torch.cat([roiSignals, roiSignals_space], dim=-1) # (batchSize, dynamicLength, 2N)

        logits = self.classifierHead(roiSignals.mean(dim=1))

        return logits


