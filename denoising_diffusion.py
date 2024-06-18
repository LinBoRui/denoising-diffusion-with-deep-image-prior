from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


if __name__ == '__main__':

    dataset_path = '/path/to/dataset'

    model = Unet(
        dim = 32,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        timesteps = 250,            # number of steps
        sampling_timesteps = 50     # number of sampling timesteps
    )

    trainer = Trainer(
        diffusion,
        dataset_path,
        train_batch_size = 50,
        train_lr = 8e-5,
        train_num_steps = 20000,          # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = True,             # whether to calculate fid during training
        results_folder = f'./results'    # folder to save results
    )

    trainer.train()
