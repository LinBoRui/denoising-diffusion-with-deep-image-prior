from denoising_diffusion_pytorch import Unet, Trainer

from deep_image_prior.models import get_net

from models import GaussianDiffusionWithDeepImagePrior


if __name__ == '__main__':
    
    set_name = 'airplane'
    dataset_path = f'./datasets/cifar10/train/{set_name}'

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

    model = Unet(
        dim = 32,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    diffusion = GaussianDiffusionWithDeepImagePrior(
        model,
        dip_model,
        dip_input_depth,
        image_size = 32,
        timesteps = 50,            # number of steps
        sampling_timesteps = 50     # number of sampling timesteps
    )

    trainer = Trainer(
        diffusion,
        dataset_path,
        train_batch_size = 50,
        train_lr = 8e-5,
        train_num_steps = 5000,          # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = True,             # whether to calculate fid during training
        results_folder = f'./results/ddpm_dip_1/{set_name}/'    # folder to save results
    )

    trainer.train()
