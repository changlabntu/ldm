import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from ldm.data_utils import imagesc

import torch.utils.data as data
import tifffile as tiff

from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.yztoxy import AETrain as Dataset
import cv2
import time


def filter_b():
    flist = sorted(glob.glob("/media/ExtHDD01/Dataset/paired_images/womac4/full/b/*.tif"))
    bmean = []
    for f in tqdm(flist):
        img = tiff.imread(f)
        bmean.append((img >= 500).sum())

    # plot histogram
    plt.hist(bmean, bins=100)
    plt.xlim(0, 2000)
    plt.show()

    for i, f in enumerate(flist):
        if bmean[i] < 500:
            # copy file to another folder
            os.system(f"cp {f} {f.replace('b', 'bclean')}")


config_name = 'womac4/2024-09-01T00-05-26_womac4x3disc'
# config_name = 'womac4/2024-08-30T06-47-48_womac4x3'

file_name = sorted(glob.glob("/media/ExtHDD01/ldmlogs/" + config_name + "/configs/*project.yaml"))[0]
config = OmegaConf.load(file_name)
model = instantiate_from_config(config.model)
model.load_state_dict(torch.load("/media/ExtHDD01/ldmlogs/" + config_name +
                                 "/checkpoints/last.ckpt", map_location="cpu")["state_dict"],strict=True)

dataset = Dataset(data_root="/media/ExtHDD01/Dataset/paired_images/womac4/full/a",
                  image_size=256,
                  trd=[0, 800], mode='test')

eval_dataloader = data.DataLoader(dataset=dataset, batch_size=1,
                                  num_workers=1, drop_last=True)

x = dataset.__getitem__(41)
print(x['file_path_'])
x = x['image']
imagesc(x.squeeze())

dec, posterior = model.forward(x.unsqueeze(0), sample_posterior=True)
imagesc(dec.detach().squeeze())