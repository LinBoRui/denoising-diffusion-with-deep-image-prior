# Denoising Diffusion Probabilistic Model with Deep Image Prior

This project aims to combine the [Denoising Diffusion Probabilistic Model (DDPM)](https://github.com/lucidrains/denoising-diffusion-pytorch) with the [Deep Image Prior (DIP)](https://github.com/DmitryUlyanov/deep-image-prior).

We know that DDPM requires a slow denoising process when sampling images. Therefore, this project attempts to incorporate the DIP model into the sampling step, aiming to produce clearer outputs with fewer steps.

## Install

- python = 3.10

```shell
pip install -r requirements.txt
```

## Implement Method

### Method 1: Integrating the DIP Model into DDPM

Attempt to integrate the DIP Model into the DDPM for joint training, so that the DIP can be trained alongside the DDPM and achieve better results.

There are currently some shortcomings, as the DIP model is unable to handle training with a large number of images, leading to unsatisfactory outcomes.

### Method 2: Using Noise Generated by the DIP Model as Training Material for DDPM

First, use the DIP Model to train on each image in the training set individually, then generate noise and save it to a file. Next, incorporate this noise into the training process.

The noise is generated using the formula:
```noise = alpha * dip_noise + (1 - alpha) * random```

However, the current results are not ideal, possibly due to issues with the noise. If a balance can be found between the noise generated by the DIP and the random noise, it might accelerate the training of the DDPM.
