#python main.py --base configs/autoencoder/autoencoder_kl_32x32x4_DPM4X.yaml -t --gpus 0,1,2,3,4,5
#python main.py --base configs/latent-diffusion/yztoxy_ori_DPM4x.yaml -t --gpus 0,1,2,3,4,5
#python main.py --base configs/latent-diffusion/yztoxy_ori_Fly0B.yaml -t --gpus 0,1,2,3 --scale_lr False
#python main.py --base configs/latent-diffusion/yztoxy_ori_Fly0B_gan.yaml -t --gpus 0,1,2,3 --scale_lr False
#python main.py --base configs/latent-diffusion/yztoxy_ori_Fly0B_gan_ae3_070924.yaml -t --gpus 0,1,2,3 --scale_lr False

#python main.py --base configs/latent-diffusion/aetesting.yaml -t --gpus 1,
python main.py --base configs/autoencoder/womac4x3.yaml -t --gpus 0,1,2,3