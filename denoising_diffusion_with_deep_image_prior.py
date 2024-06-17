from denoising_diffusion_pytorch import Unet, GaussianDiffusion

from deep_image_prior.models import get_net

from models import Trainer, DIPTrainer


if __name__ == '__main__':
    
    dataset_path = '/path/to/dataset'

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
    
    generate_noise(dip_model, dip_input_depth, dataset_path, noise_path)

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
        dataset_path,
        noise_path,
        train_batch_size = 50,
        train_lr = 8e-5,
        train_num_steps = 10000,          # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = True,             # whether to calculate fid during training
        results_folder = f'./results/ddpm_dip/{set_name}/'    # folder to save results
    )

    trainer.train()

    # sampled_images = diffusion.sample(batch_size = 4)
    # sampled_images.shape