# Denoising Diffusion Probabilistic Model with Deep Image Prior

This project aims to combine the [Denoising Diffusion Probabilistic Model (DDPM)](https://github.com/lucidrains/denoising-diffusion-pytorch) with the [Deep Image Prior (DIP)](https://github.com/DmitryUlyanov/deep-image-prior).

We know that DDPM requires a slow denoising process when sampling images. Therefore, this project attempts to incorporate the DIP model into the sampling step, aiming to produce clearer outputs with fewer steps.

## Install

- python = 3.10

```shell
pip install -r requirements.txt
```

## Implement Method

