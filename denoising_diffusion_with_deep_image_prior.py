from denoising_diffusion_pytorch import Unet, GaussianDiffusion

from deep_image_prior.models import get_net

from models import Trainer, generate_noise

import torch

if __name__ == '__main__':

    datasets_folder = '/path/to/dataset'
    noises_folder = f'{datasets_folder}/noises'

    dip_input_depth = 32
    dip_model = get_net(input_depth = dip_input_depth,
                        NET_TYPE = 'skip',
                        pad = 'reflection',
                        upsample_mode = 'bilinear',
                        skip_n33d = 128, 
                        skip_n33u = 128, 
                        skip_n11 = 4, 
                        num_scales = 3
    )
    
    generate_noise(dip_model, dip_input_depth, datasets_folder, noises_folder, train_num_steps=100)

    model = Unet(
        dim = 32,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        timesteps = 50,            # number of steps
        sampling_timesteps = 50     # number of sampling timesteps
    )

    trainer = Trainer(
        diffusion,
        datasets_folder,
        noises_folder,
        train_batch_size = 50,
        train_lr = 8e-5,
        train_num_steps = 200,          # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        calculate_fid = True,             # whether to calculate fid during training
        results_folder = f'./results'    # folder to save results
    )

    trainer.train()

    # sampled_images = diffusion.sample(batch_size = 4)
    # sampled_images.shape