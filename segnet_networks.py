# -*- coding: utf-8 -*-
"""
Created on Mon April 20 16:13:41 2020

@author: Matsui
"""


import chainer
import chainer.functions as F
import chainer.links as L


class CBRPart(chainer.Chain):
    def __init__(self, in_channel, out_channel, filter_size=3):
        super(CBRPart, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel,
                                 ksize=filter_size, pad=1),
            bnorm=L.BatchNormalization(out_channel, initial_beta=0.001)
            )

    def __call__(self, x):
        y = F.relu(self.bnorm(self.conv(x)))
        return y


class SegNetBasic(chainer.Chain):
    def __init__(self, in_channel=3, out_channel=11, c1=64, c2=128, c3=128,
                 c4=256, c5=256, filter_size1=3):
        super(SegNetBasic, self).__init__(
            # Convolution Parts
            conv1=CBRPart(in_channel, c1),
            conv2=CBRPart(c1, c2),
            conv3=CBRPart(c2, c3),
            conv4=CBRPart(c3, c4),
            conv5=CBRPart(c4, c5),

            dconv4=CBRPart(c5, c4),
            dconv3=CBRPart(c4, c3),
            dconv2=CBRPart(c3, c2),
            dconv1=CBRPart(c2, c1),
            conv_classifier=L.Convolution2D(c1, out_channel, 1, 1, 0)
            )

    def __call__(self, x):

        outsize1 = x.shape[-2:]
        h = self.conv1(x)
        h = F.max_pooling_2d(h, 2)

        outsize2 = h.shape[-2:]
        h = self.conv2(h)
        h = F.max_pooling_2d(h, 2)

        outsize3 = h.shape[-2:]
        h = self.conv3(h)
        h = F.max_pooling_2d(h, 2)

        outsize4 = h.shape[-2:]
        h = self.conv4(h)
        h = F.max_pooling_2d(h, 2)

        outsize5 = h.shape[-2:]
        h = self.conv5(h)
        h = F.max_pooling_2d(h, 2)

        h = F.unpooling_2d(h, 2, outsize=outsize5)
        h = self.dconv4(h)

        h = F.unpooling_2d(h, 2, outsize=outsize4)
        h = self.dconv3(h)

        h = F.unpooling_2d(h, 2, outsize=outsize3)
        h = self.dconv2(h)

        h = F.unpooling_2d(h, 2, outsize=outsize2)
        h = self.dconv1(h)

        h = F.unpooling_2d(h, 2, outsize=outsize1)
        y = self.conv_classifier(h)
        return y


class SegNetBasicOnly(chainer.Chain):
    def __init__(self, in_channel=3, out_channel=11, c1=48, c2=64, c3=64,
                 c4=64, c5=128, filter_size1=3):
        super(SegNetBasicOnly, self).__init__(
            # Convolution Parts
            conv1=CBRPart(in_channel, c1),
            conv2=CBRPart(c1, c2),
            conv3=CBRPart(c2, c3),
            conv4=CBRPart(c3, c4),
            conv5=CBRPart(c4, c5),

            dconv4=CBRPart(c5, c4),
            dconv3=CBRPart(c4, c3),
            dconv2=CBRPart(c3, c2),
            dconv1=CBRPart(c2, c1),
            conv_classifier=L.Convolution2D(c1, out_channel, 1, 1, 0)
            )

    def __call__(self, x):

        outsize1 = x.shape[-2:]
        h = self.conv1(x)
        h = F.max_pooling_2d(h, 2)

        outsize2 = h.shape[-2:]
        h = self.conv2(h)
        h = F.max_pooling_2d(h, 2)

        outsize3 = h.shape[-2:]
        h = self.conv3(h)
        h = F.max_pooling_2d(h, 2)

        outsize4 = h.shape[-2:]
        h = self.conv4(h)
        h = F.max_pooling_2d(h, 2)

        outsize5 = h.shape[-2:]
        h = self.conv5(h)
        h = F.max_pooling_2d(h, 2)

        h = F.unpooling_2d(h, 2, outsize=outsize5)
        h = self.dconv4(h)

        h = F.unpooling_2d(h, 2, outsize=outsize4)
        h = self.dconv3(h)

        h = F.unpooling_2d(h, 2, outsize=outsize3)
        h = self.dconv2(h)

        h = F.unpooling_2d(h, 2, outsize=outsize2)
        h = F.dropout(self.dconv1(h))

        h = F.unpooling_2d(h, 2, outsize=outsize1)
        y = self.conv_classifier(h)
        return y


class SegNetBasicAll(chainer.Chain):
    def __init__(self, in_channel=3, out_channel=11, c1=48, c2=64, c3=64,
                 c4=64, c5=128, filter_size1=3):
        super(SegNetBasicAll, self).__init__(
            # Convolution Parts
            conv1=CBRPart(in_channel, c1),
            conv2=CBRPart(c1, c2),
            conv3=CBRPart(c2, c3),
            conv4=CBRPart(c3, c4),
            conv5=CBRPart(c4, c5),

            dconv4=CBRPart(c5, c4),
            dconv3=CBRPart(c4, c3),
            dconv2=CBRPart(c3, c2),
            dconv1=CBRPart(c2, c1),
            conv_classifier=L.Convolution2D(c1, out_channel, 1, 1, 0)
            )

    def __call__(self, x):

        outsize1 = x.shape[-2:]
        h = F.dropout(self.conv1(x))
        h = F.max_pooling_2d(h, 2)

        outsize2 = h.shape[-2:]
        h = F.dropout(self.conv2(h))
        h = F.max_pooling_2d(h, 2)

        outsize3 = h.shape[-2:]
        h = F.dropout(self.conv3(h))
        h = F.max_pooling_2d(h, 2)

        outsize4 = h.shape[-2:]
        h = F.dropout(self.conv4(h))
        h = F.max_pooling_2d(h, 2)

        outsize5 = h.shape[-2:]
        h = F.dropout(self.conv5(h))
        h = F.max_pooling_2d(h, 2)

        h = F.unpooling_2d(h, 2, outsize=outsize5)
        h = F.dropout(self.dconv4(h))

        h = F.unpooling_2d(h, 2, outsize=outsize4)
        h = F.dropout(self.dconv3(h))

        h = F.unpooling_2d(h, 2, outsize=outsize3)
        h = F.dropout(self.dconv2(h))

        h = F.unpooling_2d(h, 2, outsize=outsize2)
        h = F.dropout(self.dconv1(h))

        h = F.unpooling_2d(h, 2, outsize=outsize1)
        y = self.conv_classifier(h)
        return y


if __name__ == '__main__':
    # SegNetBasic()
    # SegNetBasicOnly()
    # SegNetBasicAll()
