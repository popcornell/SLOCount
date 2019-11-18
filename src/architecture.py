# Modified from https://github.com/kaituoxu/Conv-TasNet/tree/master/src

import torch.nn as nn
import torch.nn.functional as F
import torch
from src.custom_layers import chose_norm, ChannelwiseLayerNorm, HSwish, DepthwiseSeparableConv


class TemporalConvNet(nn.Module):
    def __init__(self, N,  B, H, P, X, R, C=5, norm_type="BN",
                 causal=False, multi_channel=32):
        """
        Args:
            N: Number of filters input to the network
            N_out : Number of bins in the mask
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(TemporalConvNet, self).__init__()
        # Hyper-parameter
        self.N = N
        self.C = C
        self.X = X
        self.R = R
        self.multi_channel = multi_channel

        # Components
        # [M, N, K] -> [M, N, K]

        self.layer_norm_sc = ChannelwiseLayerNorm(N-181)

        # [M, N, K] -> [M, B, K]
        self.bottleneck_conv1x1_sc = nn.Conv1d(N-181, B-multi_channel, 1)

        if self.multi_channel >0:
            self.layer_norm_multi = ChannelwiseLayerNorm(181)
            self.bottleneck_conv1x1_multi = nn.Conv1d(181, multi_channel, 1)
        # [M, B, K] -> [M, B, K]
        self.TCN = nn.ModuleList()
        for r in range(R):
            for x in range(X):
                dilation = 2**x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                self.TCN.append(TemporalBlock(B, H, P, stride=1,
                                              padding=padding,
                                              dilation=dilation,
                                              norm_type=norm_type,
                                              causal=causal))
        # [M, B, K] -> [M, C*N, K]

        mask_conv1x1 = nn.Conv1d(B, C, 1)

        # Put together
        #self.BN = nn.Sequential(layer_norm, bottleneck_conv1x1)
        self.mask_net = nn.Sequential(HSwish(), mask_conv1x1)

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """

        if self.multi_channel>0:
            sc = mixture_w[:, :-181, :]
            gcc = mixture_w[: , -181:, :]
            sc = self.layer_norm_sc(sc)
            gcc = self.layer_norm_multi(gcc)
            sc = self.bottleneck_conv1x1_sc(sc)
            gcc = self.bottleneck_conv1x1_multi(gcc)

            output = torch.cat([sc, gcc], 1)

        else:
            sc = self.layer_norm(mixture_w)
            output = self.bottleneck_conv1x1_sc(sc)


        M, N, K = output.size()
        #output = self.BN(mixture_w)


        skip_connection = 0.
        for i in range(len(self.TCN)):
            residual, skip = self.TCN[i](output)
            output = output + residual
            skip_connection = skip_connection + skip
        # It would make sense to concat the residual and the skip, to try maybe.
        score = self.mask_net(skip_connection)

        score = score.view(M, self.C,  K) # [M, C*N, K] -> [M, C, N, K]
        est_mask = F.log_softmax(score,  1)
        return est_mask


    def get_fov(self, L):
        """
        :return: receptive field in samples
        """
        fov = L
        for x in range(self.X):
            fov += self.R * (L) * 2 ** x
        return fov


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="BN", causal=False):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        act = HSwish()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation, norm_type,
                                        causal)
        # Put together
        self.net = nn.Sequential(conv1x1,  norm,  act, dsconv)
        self.res_conv = nn.Conv1d(out_channels, in_channels, 1)
        self.skip_conv = nn.Conv1d(out_channels, in_channels, 1)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K], [M, B, K] (residual, skip)
        """
        out = self.net(x)
        return self.res_conv(out), self.skip_conv(out)



def get_SLOCountNet(hp):

    return TemporalConvNet(hp.model.n_input_channels, hp.model.n_filters_b, hp.model.n_filters_c,
                           hp.model.ksz, hp.model.n_blocks, hp.model.n_repeats, multi_channel=hp.features.n_gcc)



