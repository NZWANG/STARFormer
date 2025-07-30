import torch
from torch import nn

from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

from Models.STARFormer.util import windowBoldSignal

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 1,
        dropout = 0.,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim
        activation = nn.GELU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            activation
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class WindowAttention(nn.Module):

    def __init__(self, dim, windowSize, receptiveSize, numHeads, headDim=20, attentionBias=True, qkvBias=True, attnDrop=0., projDrop=0.):

        super().__init__()
        self.dim = dim
        self.windowSize = windowSize  # N
        self.receptiveSize = receptiveSize # M
        self.numHeads = numHeads
        head_dim = headDim
        self.scale = head_dim ** -0.5

        self.attentionBias = attentionBias

        # define a parameter table of relative position bias
        maxDisparity = windowSize + (receptiveSize - windowSize)//2


        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2*maxDisparity, numHeads))  # maxDisparity, nH

        # get pair-wise relative position index for each token inside the window
        coords_x = torch.arange(self.windowSize) # N
        coords_x_ = torch.arange(self.receptiveSize) - (self.receptiveSize - self.windowSize)//2 # M
        relative_coords = coords_x[:, None] - coords_x_[None, :]  # N, M
        relative_coords[:, :] += maxDisparity  # shift to start from 0
        relative_position_index = relative_coords  # (N, M)
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, head_dim * numHeads, bias=qkvBias)
        self.kv = nn.Linear(dim, 2 * head_dim * numHeads, bias=qkvBias)

        self.attnDrop = nn.Dropout(attnDrop)
        self.proj = nn.Linear(head_dim * numHeads, dim)

        self.projDrop = nn.Dropout(projDrop)

        # prep the biases
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
        self.softmax = nn.Softmax(dim=-1)

        self.attentionMaps = None # shape = (#windows * nH, 1+windowSize, 1+receptiveSize)
        self.attentionGradients = None # shape = (#windows * nH, 1+windowSize, 1+receptiveSize)
        self.nW = None

    def save_attention_maps(self, attentionMaps):
        self.attentionMaps = attentionMaps

    def save_attention_gradients(self, grads):
        self.attentionGradients = grads


    def forward(self, x, x_, mask, nW):

        B_, N, C = x.shape # (B*num_windows, windowSize, dim)
        _, M, _ = x_.shape # (B*num_windows, receptiveSize, dim)

        B = B_ // nW # batchsize

        mask_left, mask_right = mask

        # linear mapping
        q = self.q(x) # (batchSize * #windows, windowSize, numheads*head_dim)
        k, v = self.kv(x_).chunk(2, dim=-1) # (batchSize * #windows, receptiveSize, 2*numheads*head_dim)

        # head seperation
        q = rearrange(q, "b n (h d) -> b h n d", h=self.numHeads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.numHeads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.numHeads)

        attn = torch.matmul(q , k.transpose(-1, -2)) * self.scale # (batchSize*#windows, h, n, m)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, M, -1)  # N, M, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, M

        if(self.attentionBias):
            attn[:, :, :, :] = attn[:, :, :, :] + relative_position_bias.unsqueeze(0)

        # mask the not matching queries and tokens here
        maskCount = mask_left.shape[0]
        # repate masks for batch and heads
        mask_left = repeat(mask_left, "nM nn mm -> b nM h nn mm", b=B, h=self.numHeads) # (batchsize, nummask, numheads, windowSize, receptiveSize)
        mask_right = repeat(mask_right, "nM nn mm -> b nM h nn mm", b=B, h=self.numHeads) # (batchsize, nummask, numheads, windowSize, receptiveSize)

        mask_value = max_neg_value(attn) 


        attn = rearrange(attn, "(b nW) h n m -> b nW h n m", nW = nW)        
        
        # make sure masks do not overflow
        maskCount = min(maskCount, attn.shape[1])
        mask_left = mask_left[:, :maskCount]
        mask_right = mask_right[:, -maskCount:]

        attn[:, :maskCount].masked_fill_(mask_left==1, mask_value)
        attn[:, -maskCount:].masked_fill_(mask_right==1, mask_value)
        attn = rearrange(attn, "b nW h n m -> (b nW) h n m")

        attn = self.softmax(attn) # (b, h, n, m)

        attn = self.attnDrop(attn)

        x = torch.matmul(attn, v) # of shape (b_, h, n, d)

        x = rearrange(x, 'b h n d -> b n (h d)')

        x = self.proj(x)
        x = self.projDrop(x)
        
        return x

class FusedWindowTransformer(nn.Module):

    def __init__(self, dim, windowSize, shiftSize, receptiveSize, numHeads, headDim, mlpRatio, attentionBias, drop, attnDrop):
        
        super().__init__()

        self.attention = WindowAttention(dim=dim, windowSize=windowSize, receptiveSize=receptiveSize, numHeads=numHeads, headDim=headDim, attentionBias=attentionBias, attnDrop=attnDrop, projDrop=drop)
        
        self.mlp = FeedForward(dim=dim, mult=mlpRatio, dropout=drop)

        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)

        self.shiftSize = shiftSize

    def forward(self, x, windowX, windowX_, mask, nW):

        windowXTrans = self.attention(self.attn_norm(windowX), self.attn_norm(windowX_), mask, nW) # (B*nW, windowSize, C)
        xTrans = windowXTrans # (B*nW, windowSize, C)

        xTrans = rearrange(xTrans, "(b nW) l c -> b nW l c", nW=nW)

        xTrans = self.gatherWindows(xTrans, x.shape[1], self.shiftSize)
        
        # residual connections
        xTrans = xTrans + x

        # MLP layers
        xTrans = xTrans + self.mlp(self.mlp_norm(xTrans))

        return xTrans

    def gatherWindows(self, windowedX, dynamicLength, shiftSize):

        batchSize = windowedX.shape[0]
        windowLength = windowedX.shape[2]
        nW = windowedX.shape[1]
        C = windowedX.shape[-1]
        
        device = windowedX.device

        destination = torch.zeros((batchSize, dynamicLength,  C)).to(device)
        scalerDestination = torch.zeros((batchSize, dynamicLength, C)).to(device)

        indexes = torch.tensor([[j+(i*shiftSize) for j in range(windowLength)] for i in range(nW)]).to(device)
        indexes = indexes[None, :, :, None].repeat((batchSize, 1, 1, C)) # (batchSize, nW, windowSize, featureDim)

        src = rearrange(windowedX, "b n w c -> b (n w) c")
        indexes = rearrange(indexes, "b n w c -> b (n w) c")

        destination.scatter_add_(dim=1, index=indexes, src=src)

        scalerSrc = torch.ones((windowLength)).to(device)[None, None, :, None].repeat(batchSize, nW, 1, C) # (batchSize, nW, windowLength, featureDim)
        scalerSrc = rearrange(scalerSrc, "b n w c -> b (n w) c")

        scalerDestination.scatter_add_(dim=1, index=indexes, src=scalerSrc)

        destination = destination / scalerDestination

        return destination

class TransformerBlock(nn.Module):

    def __init__(self, dim, numHeads, headDim, windowSize, receptiveSize, shiftSize, mlpRatio=1.0, drop=0.0, attnDrop=0.0, attentionBias=True):

        assert((receptiveSize-windowSize)%2 == 0)

        super().__init__()
        self.transformer = FusedWindowTransformer(dim=dim, windowSize=windowSize, shiftSize=shiftSize, receptiveSize=receptiveSize, numHeads=numHeads, headDim=headDim, mlpRatio=mlpRatio, attentionBias=attentionBias, drop=drop, attnDrop=attnDrop)

        self.windowSize = windowSize
        self.receptiveSize = receptiveSize
        self.shiftSize = shiftSize

        self.remainder = (self.receptiveSize - self.windowSize) // 2

        # create mask here for non matching query and key pairs
        maskCount = self.remainder // shiftSize + 1
        mask_left = torch.zeros(maskCount, self.windowSize, self.receptiveSize)
        mask_right = torch.zeros(maskCount, self.windowSize, self.receptiveSize)

        for i in range(maskCount):
            if(self.remainder > 0):
                mask_left[i, :, :self.remainder-shiftSize*i] = 1
                if((-self.remainder+shiftSize*i) < 0):
                    mask_right[maskCount-1-i, :, -self.remainder+shiftSize*i:] = 1

        self.register_buffer("mask_left", mask_left)
        self.register_buffer("mask_right", mask_right)
    
    def forward(self, x):

        B, Z, C = x.shape
        device = x.device

        x = x[:, :Z]

        # form the padded x to be used for focal keys and values
        x_ = torch.cat([torch.zeros((B, self.remainder,C),device=device), x, torch.zeros((B, self.remainder,C), device=device)], dim=1) # (B, remainder+Z+remainder, C) 

        windowedX, _ = windowBoldSignal(x.transpose(2,1), self.windowSize, self.shiftSize) # (B, nW, C, windowSize)         
        windowedX = windowedX.transpose(2,3) # (B, nW, windowSize, C)

        windowedX_, _ = windowBoldSignal(x_.transpose(2,1), self.receptiveSize, self.shiftSize) # (B, nW, C, receptiveSize)
        windowedX_ = windowedX_.transpose(2,3) # (B, nW, receptiveSize, C)
        
        nW = windowedX.shape[1] # number of windows

        windowedX = rearrange(windowedX, "b nw l c -> (b nw) l c") # (B*nW, windowSize, C)
        windowedX_ = rearrange(windowedX_, "b nw l c -> (b nw) l c") # (B*nW, receptiveSize, C)

        masks = [self.mask_left, self.mask_right]

        fusedX_trans = self.transformer(x, windowedX, windowedX_, masks, nW) # (B*nW, windowSize, C)

        return fusedX_trans
