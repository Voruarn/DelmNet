import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Modules import *
from .init_weights import init_weights
from .ConvNextV2 import convnextv2_tiny, convnextv2_small, convnextv2_base
from .ResNet import resnet50
from .pvt import pvt_tiny, pvt_small


class DelmNet(nn.Module):
    # Detail Enhancement Location Mining Network
    # backbone:  convnextv2_tiny, resnet50, pvt_small
    def __init__(self, backbone='convnextv2_tiny', mid_ch=128, bottleneck_num=2, **kwargs):
        super(DelmNet, self).__init__()      

        self.encoder=eval(backbone)()
        enc_dims=[96, 192, 384, 768]
        if backbone=='resnet50':
            enc_dims=[256, 512, 1024, 2048]  
        elif backbone=='pvt_small':
            enc_dims=[64, 128, 320, 512]  

        out_ch=1
        # Encoder
        self.eside1=ConvModule(enc_dims[0], mid_ch)
        self.eside2=ConvModule(enc_dims[1], mid_ch)
        self.eside3=ConvModule(enc_dims[2], mid_ch)
        self.eside4=ConvModule(enc_dims[3], mid_ch)

        # Decoder
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.lmm1=LMM1(mid_ch)
        self.lmm2=LMM(mid_ch)
        self.lmm3=LMM(mid_ch)
        self.lmm4=LMM4(mid_ch)

        self.dem=DEM(mid_ch, mid_ch)

        self.dec1=Decoder(c1=mid_ch, c2=mid_ch, n=bottleneck_num)
        self.dec2=Decoder(c1=mid_ch, c2=mid_ch, n=bottleneck_num)
        self.dec3=Decoder(c1=mid_ch, c2=mid_ch, n=bottleneck_num)
        self.dec4=Decoder(c1=mid_ch, c2=mid_ch, n=bottleneck_num)

        self.dside1 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside2 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside3 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside4 = nn.Conv2d(mid_ch,out_ch,3,padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        # encoder
        outs = self.encoder(inputs)
        c1, c2, c3, c4 = outs
    
        c1=self.eside1(c1)
        c2=self.eside2(c2)
        c3=self.eside3(c3)
        c4=self.eside4(c4)

        cc1,cc2,cc3,cc4=self.dem(c1,c2,c3,c4)
        # Feedback from dem
        cc12=F.interpolate(cc1, size=cc2.size()[2:], mode='bilinear', align_corners=True)
        cc13=F.interpolate(cc1, size=cc3.size()[2:], mode='bilinear', align_corners=True)
        cc14=F.interpolate(cc1, size=cc4.size()[2:], mode='bilinear', align_corners=True)

        # LMM
        ca1=self.lmm1(c1,c2)
        ca2=self.lmm2(c1,c2,c3)
        ca3=self.lmm3(c2,c3,c4)
        ca4=self.lmm4(c3,c4)

        # Decoder
        up4=ca4 + c4 + cc14
        up4=self.dec4(up4)

        up3=self.upsample2(up4) + ca3 + cc13
        up3=self.dec3(up3)

        up2=self.upsample2(up3) + ca2 + cc12
        up2=self.dec2(up2)

        up1=self.upsample2(up2) + ca1 + cc1
        up1=self.dec1(up1)

        d1=self.dside1(up1)
        d2=self.dside2(up2)
        d3=self.dside3(up3)
        d4=self.dside4(up4)
      
        S1 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)
        S2 = F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)
        S3 = F.interpolate(d3, size=(H, W), mode='bilinear', align_corners=True)
        S4 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=True)

        return S1,S2,S3,S4, torch.sigmoid(S1),torch.sigmoid(S2),torch.sigmoid(S3),torch.sigmoid(S4)

