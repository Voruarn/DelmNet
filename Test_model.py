import torch
import torch.nn.functional as F

import numpy as np
# from fcmae_model import convnextv2_nano, convnextv2_tiny, convnextv2_base
from network.DelmNet import DelmNet
from SSMAE import ssmae

if __name__=='__main__':
    print('Test DelmNet_X')

    input=torch.rand(2,3,256,256).cuda()
    # mask=torch.rand(2,1,256,256).cuda()

    model = DelmNet().cuda()
    model.eval()
    output=model(input)

    # model = ssmae().cuda()
    # model.eval()
    # output=model(input, mask)

    for x in output:
        print('x.shape:',x.shape)

