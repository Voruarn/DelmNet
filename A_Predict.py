import torch
import torch.nn.functional as F

import numpy as np
import os, argparse
import imageio
import time

from network.DelmNet import DelmNet
from data import test_dataset
import collections
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--testset_path", type=str, 
        default='../Dataset/EORSSD/Test/',
        help="path to Dataset")
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument("--ckpt", type=str,
            default='',
              help="restore from checkpoint")
parser.add_argument("--pred_path", type=str, default='',
                        help="random seed (default: 1)")

opt = parser.parse_args()

if not os.path.exists(opt.pred_path):
    os.makedirs(opt.pred_path)
    
model = DelmNet()
if opt.ckpt is not None and os.path.isfile(opt.ckpt):
    checkpoint = torch.load(opt.ckpt, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(checkpoint["model_state"])
        print('try: load pth from:', opt.ckpt)
    except:
        dic = collections.OrderedDict()
        for k, v in checkpoint["model_state"].items():
            #print( k)
            mlen=len('module')+1
            newk=k[mlen:]
            # print(newk)
            dic[newk]=v
        model.load_state_dict(dic)
        print('except: load pth from:', opt.ckpt)

else:
    print("[!] Retrain")
     
model.cuda()
model.eval()


#test_datasets = ['EORSSD','ORSSD','ors-4199']

print('start pred...')

image_root = opt.testset_path  + 'Images/'

gt_root = opt.testset_path +'Masks/'
test_loader = test_dataset(image_root, gt_root, opt.testsize)
time_sum = 0
for i in tqdm(range(test_loader.size)):
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()
    time_start = time.time()
    s1,s2,s3,s4, s1_sig,s2_sig,s3_sig,s4_sig = model(image)
    res=s1
    time_end = time.time()
    time_sum = time_sum+(time_end-time_start)
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    imageio.imsave(opt.pred_path+name, res)
    if i == test_loader.size-1:
        print('Running time {:.5f}'.format(time_sum/test_loader.size))
        print('FPS {:.5f}'.format(test_loader.size / time_sum))

