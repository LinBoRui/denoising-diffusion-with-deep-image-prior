from random import random
import math
from multiprocessing import cpu_count

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import utils

from einops import rearrange, reduce

from tqdm import tqdm

from denoising_diffusion_pytorch import GaussianDiffusion, Trainer

from deep_image_prior.utils.denoising_utils import *


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def cycle(dl):
    while True:
        for idx, data in enumerate(dl):
            yield idx, data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def normalize_norm_dist(x):
    return (x - torch.mean(x)) / torch.std(x)


class GaussianDiffusionWithDeepImagePrior(GaussianDiffusion):
    def __init__(
        self,
        base_model,
        dip_model,
        dip_input_depth,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__(
            model = base_model,
            image_size = image_size,
            timesteps = timesteps,
            sampling_timesteps = sampling_timesteps,
            objective = objective,
            beta_schedule = beta_schedule,
            schedule_fn_kwargs = schedule_fn_kwargs,
            ddim_sampling_eta = ddim_sampling_eta,
            auto_normalize = auto_normalize,
            offset_noise_strength = offset_noise_strength,
            min_snr_loss_weight = min_snr_loss_weight,
            min_snr_gamma = min_snr_gamma
        )
        
        self.dip_model = dip_model.type(torch.FloatTensor)
        self.dip_input_depth = dip_input_depth
        
    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        dip_input = get_noise(self.dip_input_depth, 'noise', shape[-2:]).to(device)
        dip_input = dip_input.expand(batch, -1, -1, -1)
        img = self.dip_model(dip_input)
        img = normalize_norm_dist(img)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps, leave = False):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        dip_input = get_noise(self.dip_input_depth, 'noise', shape[-2:]).to(device)
        dip_input = dip_input.expand(batch, -1, -1, -1)
        img = self.dip_model(dip_input)
        img = normalize_norm_dist(img)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', leave = False):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape
        
        dip_input = get_noise(self.dip_input_depth, 'noise', (h, w)).to(self.device)
        dip_input = dip_input.expand(b, -1, -1, -1)
        
        dip_out = self.dip_model(dip_input)
        
        noise = default(noise, lambda: normalize_norm_dist(dip_out - x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        
        dip_loss = F.mse_loss(dip_out, x_start)
        dip_loss = dip_loss.mean() * self.loss_weight[-1]
        
        return loss.mean() + dip_loss

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)


class Trainer(Trainer):
    def __init__(
        self,
        diffusion_model,
        dip_model,
        dip_input_depth,
        folder,
        *,
        dip_steps = 100,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False
    ):
        super().__init__(
            diffusion_model,
            folder,
            train_batch_size = train_batch_size,
            gradient_accumulate_every = gradient_accumulate_every,
            augment_horizontal_flip = augment_horizontal_flip,
            train_lr = train_lr,
            train_num_steps = train_num_steps,
            ema_update_every = ema_update_every,
            ema_decay = ema_decay,
            adam_betas = adam_betas,
            save_and_sample_every = save_and_sample_every,
            num_samples = num_samples,
            results_folder = results_folder,
            amp = amp,
            mixed_precision_type = mixed_precision_type,
            split_batches = split_batches,
            convert_image_to = convert_image_to,
            calculate_fid = calculate_fid,
            inception_block_idx = inception_block_idx,
            max_grad_norm = max_grad_norm,
            num_fid_samples = num_fid_samples,
            save_best_and_latest_only = save_best_and_latest_only
        )
        
        self.dip_model = dip_model
        self.dip_input_depth = dip_input_depth
        self.dip_steps = dip_steps
        self.dip_folder = self.results_folder / 'dip'
        
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.noise = torch.tensor([], device=self.accelerator.device)
        for idx, data in tqdm(enumerate(dl), desc='generating noise'):
            noise = self.get_noise(idx, data).unsqueeze(0)
            self.noise = torch.cat((self.noise, noise), dim = 0)

    def generate_noise(self, data):
        noise = torch.tensor([], device=self.accelerator.device)
        for d in tqdm(data, leave=False):
            d = d.unsqueeze(0)
            dip_trainer = DIPTrainer(
                model = self.dip_model,
                dip_input_depth = self.dip_input_depth,
                train_img = d,
                results_folder = self.dip_folder,
                device = self.accelerator.device
            )
            dip_trainer.train(train_num_steps = self.dip_steps)
            noise = torch.cat((noise, dip_trainer.generate_noise()), dim = 0)
        return noise.clone()
    
    def get_noise(self, idx, data):
        noise_path = self.dip_folder / f'noise_{idx}.pth'
        if noise_path.exists():
            return torch.load(noise_path)
        noise = self.generate_noise(data)
        torch.save(noise, noise_path)
        return noise

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        self.noise = self.noise.to(device)

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    idx, data = next(self.dl)
                    data = data.to(device)
                    noise = self.noise[idx]

                    with self.accelerator.autocast():
                        loss = self.model(data, noise = noise)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')



class DIPTrainer:
    def __init__(
        self,
        model,
        dip_input_depth,
        train_img,
        *,
        learning_rate = 1e-4,
        adam_betas = (0.9, 0.99),
        results_folder = './results',
        device = None
    ):
        self.device = default(device, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dtype = torch.FloatTensor
        
        self.model = model.type(self.dtype).to(self.device)
        self.dip_input_depth = dip_input_depth
        self.train_img = self._load_image(train_img).type(self.dtype).to(self.device)
        self.result_folder = results_folder
        self.image_size = self.train_img.shape[-2:]

        # optimizer

        self.opt = Adam(self.model.parameters(), lr = learning_rate, betas = adam_betas)
        
        self.model_input = get_noise(self.dip_input_depth, 'noise', self.image_size).to(self.device)

        self.step = 0
        os.makedirs(self.result_folder, exist_ok=True)
    
    def _load_image(self, train_img):
        if isinstance(train_img, str):
            train_img = get_image(train_img)[0]
        if isinstance(train_img, Image.Image):
            train_img = crop_image(train_img, d=self.dip_input_depth)
            train_img = pil_to_np(train_img)
        if isinstance(train_img, np.ndarray):
            train_img = np_to_torch(train_img)
        if not isinstance(train_img, torch.Tensor):
            raise ValueError('train_img must be a string, image, numpy array or torch tensor')
        return train_img
    
    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'model_input': self.model_input,
        }
        torch.save(data, os.path.join(self.result_folder, f'dip_model_{milestone}.pth'))
    
    def load(self, milestone):
        data = torch.load(os.path.join(self.result_folder, f'dip_model_{milestone}.pth'))
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.model_input = data['model_input']
        
    def predict(self):
        self.model.eval()
        out = self.model(self.model_input)
        return out

    def show_image(self):
        out = self.predict()
        out_np = torch_to_np(out)
        img_np = torch_to_np(self.train_img)
        plot_image_grid([img_np, out_np], factor = 13)
    
    def save_image(self):
        out = self.predict()
        out_np = torch_to_np(out)
        out_img = np_to_pil(out_np)
        out_img.save(os.path.join(self.result_folder, f'dip_{self.step}.png'))
    
    def generate_noise(self):
        out = self.predict()
        noise = normalize_norm_dist(out - self.train_img)
        return noise
    
    def show_noise(self):
        noise = self.generate_noise()
        noise = (noise + 1) / 2
        noise_np = torch_to_np(noise)
        noise_np = np.clip(noise_np, 0, 1)
        plot_image_grid([noise_np], factor = 13)
    
    def train(self, train_num_steps = 10000, predict_every = 10000, save_every = 10000):
        
        with tqdm(initial = self.step, total = train_num_steps, leave = False) as pbar:
            while self.step < train_num_steps:
                self.model.train()
                self.opt.zero_grad()
                output = self.model(self.model_input)
                loss = F.mse_loss(output, self.train_img)
                loss.backward()
                self.opt.step()
                
                self.step += 1
                pbar.update(1)
                
                if self.step % predict_every == 0:
                    self.save_image()
            
                if self.step % save_every == 0:
                    milestone = self.step // save_every
                    self.save(milestone)

