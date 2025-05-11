from model.semseg.base import BaseNet
import model.backbone.resnet_ds as resnet_ds
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV3Plus(BaseNet):
    def __init__(self, backbone, nclass):
        super(DeepLabV3Plus, self).__init__(backbone)

        low_level_channels = self.backbone.channels[0]
        high_level_channels = self.backbone.channels[-1]

        self.head = ASPPModule(high_level_channels, (12, 24, 36))

        self.reduce = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_level_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),

                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))

        self.classifier = nn.Conv2d(256, nclass, 1, bias=True)

    def base_forward(self, x):
        h, w = x.shape[-2:]

        c1, _, _, c4 = self.backbone.base_forward(x)

        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        out = torch.cat([c1, c4], dim=1)
        out = self.fuse(out)

        out = self.classifier(out)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True),
                                     nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
    
def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, h*w)
    x_t =  x.permute(0,2,1)
    gram = torch.bmm(x,x_t) # 행렬간 곱셈 수행
    return gram


class DeepLabV3Plus_Aux(nn.Module):
    def __init__(self, backbone, dilations, nclass):
        super(DeepLabV3Plus_Aux, self).__init__()

        if 'resnet' in backbone:
            self.backbone = resnet_ds.__dict__[backbone](pretrained=True, replace_stride_with_dilation= [False, False, True])
        else:
            assert backbone == 'xception'
            self.backbone = xception(pretrained=True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, dilations)

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))
        
        self.fuse1 = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 304, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(304),
                                  nn.ReLU(True),
                                  nn.Conv2d(304, 304, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(304),
                                  nn.ReLU(True))
        
        
        self.conv_g = nn.Sequential(nn.Conv2d(1, 8, 3, stride=4, padding=1, bias=False),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(True),
                                   nn.Conv2d(8, 1, 3, stride=4, padding=1, bias=False),
                                   nn.ReLU(True),)


        self.classifier = nn.Conv2d(256, nclass, 1, bias=True)
        self.classifier_g = nn.Linear(256, 2)

    def forward(self, x, need_fp=False):
        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]
        

        if need_fp:
            outs, feature = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            
            gram_feature = gram_matrix(feature).unsqueeze(1)
            gram_feature = self.conv_g(gram_feature)
            gram_feature, gram_feature_fp = gram_feature.chunk(2)

            return out, out_fp, gram_feature, gram_feature_fp 

        out, feature = self._decode(c1, c4)
        
        gram_feature = gram_matrix(feature).unsqueeze(1)
        gram_feature = self.conv_g(gram_feature)
        
        
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out, gram_feature

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        out = self.classifier(feature)

        return out, feature
