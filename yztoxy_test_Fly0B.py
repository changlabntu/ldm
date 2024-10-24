import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

import torch.utils.data as data
import tifffile as tiff

from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.yztoxynorescale import ZEnhanceDataset
import cv2
import time


def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    # !!!
    #yz_dataset = ZEnhanceDataset(data_root=['/media/ExtHDD01/Dataset/paired_images/Fly0B/cycout/yzoridsp/'], data_len=[0,50],
    #yz_dataset = ZEnhanceDataset(data_root=['/media/ExtHDD01/Dataset/paired_images/Fly0B/cycout/ganori/'], data_len=[0, 64],
    yz_dataset = ZEnhanceDataset(data_root=['/media/ExtHDD01/Dataset/paired_images/Fly0B/diffout/diffroi0norm/'], data_len=[0, 256],
                                    mask_config={"direction": "horizontal", "down_size": 8}, # this is not being used
                                    #mask_type="downsample",
                                    image_size=256, mode='test')
    eval_dataloader = data.DataLoader(dataset=yz_dataset, batch_size=1,
                                       num_workers=10, drop_last=True)

    # !!!
    #config = OmegaConf.load("logs/2024-04-27T11-59-39_yztoxy_ori/configs/2024-04-27T11-59-39-project.yaml")
    if 0: # GAN 0
        config = OmegaConf.load("/media/ExtHDD01/ldmlogs/Fly0B/2024-07-09T12-12-55_yztoxy_ori_Fly0B_gan_ae3_070924/configs/2024-07-09T12-12-55-project.yaml")
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load("/media/ExtHDD01/ldmlogs/Fly0B/"
                                         "2024-07-09T12-12-55_yztoxy_ori_Fly0B_gan_ae3_070924/checkpoints/epoch=000943.ckpt")["state_dict"],
                              strict=True)
    elif 0: #DSP
        config = OmegaConf.load("/media/ExtHDD01/ldmlogs/Fly0B/2024-06-20T18-00-21_yztoxy_ori_Fly0B_dsp_ae3_no_lr_scale/configs/2024-06-20T18-00-21-project.yaml")
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load("/media/ExtHDD01/ldmlogs/Fly0B/"
                                         "2024-06-20T18-00-21_yztoxy_ori_Fly0B_dsp_ae3_no_lr_scale/checkpoints/epoch=001805.ckpt")["state_dict"],
                              strict=True)
    elif 0: # GAN x2
        config = OmegaConf.load("/media/ExtHDD01/ldmlogs/Fly0B/2024-08-13T16-08-51_yztoxy_ori_Fly0B_gan_ae3x2/configs/2024-08-13T16-08-51-project.yaml")
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load("/media/ExtHDD01/ldmlogs/Fly0B/"
                                         "2024-08-13T16-08-51_yztoxy_ori_Fly0B_gan_ae3x2/checkpoints/epoch=000661.ckpt")["state_dict"],
                              strict=True)
    elif 1: # GAN x
        config = OmegaConf.load("/media/ExtHDD01/ldmlogs/Fly0B/2024-09-01T13-07-14_test0901/configs/project.yaml")
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load("/media/ExtHDD01/ldmlogs/Fly0B/"
                                         "2024-09-01T13-07-14_test0901/checkpoints/epoch=004109.ckpt")["state_dict"],
                              strict=True)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu") # why using cpu??
    model = model.to(device).eval()
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    s_t = time.time()
    with torch.no_grad():
        with model.ema_scope():
            for ret in tqdm(eval_dataloader):
                image = ret['image'].to(device=device)
                cond_image = ret['cond_image'].to(device=device)
                batch = {"image": image, "cond_image": cond_image}
                
                # model.first_stage_model = model.first_stage_model.train()
                # model.cond_stage_model = model.cond_stage_model.train()
                
                c = model.get_learned_conditioning(batch["cond_image"])
                recon = model.cond_stage_model.decode(c)
                
                encoder_posterior = model.encode_first_stage(batch["image"])
                img_feature = model.get_first_stage_encoding(encoder_posterior).detach()
                #img_feature = encoder_posterior.detach()

                shape = img_feature.shape[1:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)#,
                                                 #x0=img_feature)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
 
                # image = torch.clamp((batch["image"]+1.0)/2.0, min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                # recon = torch.clamp((recon+1.0)/2.0, min=0.0, max=1.0)
                
                # original out
                #outpath = os.path.join(opt.outdir, ret['file_path_'][0].replace('.tif', '.tif'))
                #predicted_image = (predicted_image.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
                #tiff.imwrite(outpath, predicted_image)
                outpath = os.path.join(opt.outdir, ret['file_path_'][0].replace('.tif', '.tif'))
                predicted_image = (predicted_image.squeeze(0).detach().cpu().numpy())

                cond_image = cond_image.detach().cpu().numpy()[0, ::]

                # norm by mean and std
                (predicted_image, cond_image) = (x - x.mean() for x in (predicted_image, cond_image))
                (predicted_image, cond_image) = (x / x.std() for x in (predicted_image, cond_image))

                combined = np.concatenate([predicted_image, cond_image], 2)

                tiff.imwrite(outpath, combined)

                

                #cv2.imwrite(outpath.replace('.png', f'_pred.png'), np.transpose(np.concatenate([predicted_image, predicted_image, predicted_image], 0), (1, 2, 0)))

                # recon = (recon.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
                # cv2.imwrite(outpath.replace('.png', f'_recon.png'), np.transpose(np.concatenate([recon, recon, recon], 0), (1, 2, 0)))

                # recon = model.decode_first_stage(img_feature)
                # recon = torch.clamp((recon+1.0)/2.0,
                #                               min=0.0, max=1.0)
                # recon = (recon.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
                # cv2.imwrite(outpath.replace('.png', f'_recon_ori.png'), np.transpose(np.concatenate([recon, recon, recon], 0), (1, 2, 0)))

                # img = batch["image"]
                # img = (img + 1) / 2
                # img = (img.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
                # cv2.imwrite(outpath.replace('.png', f'_img.png'), np.transpose(np.concatenate([img, img, img], 0), (1, 2, 0)))
    e_t = time.time()
    print('Total time:', e_t-s_t)

# CUDA_VISIBLE_DEVICES=3 python inpaint.py --indir x --outdir /home/ziyi/Projects/latent-diffusion/1225_out
# CUDA_VISIBLE_DEVICES=0 python yztoxy_test_Fly0B.py --outdir /media/ExtHDD01/Dataset/paired_images/Fly0B/yzori8gan/