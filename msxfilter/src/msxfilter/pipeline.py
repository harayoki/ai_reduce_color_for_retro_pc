"""Two-stage pipeline: color (A) then pixel-ish (B)."""
from __future__ import annotations

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from .rules import palette_loss_lab, rgb01_to_lab, tile_proxy_loss


def _encode_latents(pipe: StableDiffusionPipeline, img01_hwc: torch.Tensor, device: torch.device) -> torch.Tensor:
    t = img01_hwc.permute(2, 0, 1).unsqueeze(0).to(device)
    t = t * 2 - 1
    with torch.no_grad():
        z = pipe.vae.encode(t).latent_dist.sample()
        z = z * pipe.vae.config.scaling_factor
    return z


def _decode_latents(pipe: StableDiffusionPipeline, latents: torch.Tensor) -> torch.Tensor:
    z = latents / pipe.vae.config.scaling_factor
    img = pipe.vae.decode(z).sample
    return (img / 2 + 0.5).clamp(0, 1)  # (1,3,H,W)


def _img2img_init(pipe: StableDiffusionPipeline, base_latents: torch.Tensor, steps: int, denoise: float) -> tuple[torch.Tensor, int]:
    pipe.scheduler.set_timesteps(steps, device=base_latents.device)
    start = int(steps * denoise)
    start = max(0, min(steps - 1, start))
    t0 = pipe.scheduler.timesteps[start]
    return pipe.scheduler.add_noise(base_latents, torch.randn_like(base_latents), t0), start


def _warmup_weight(alpha: float, i: int, total: int, warmup_ratio: float) -> float:
    if total <= 1:
        return alpha
    p = i / (total - 1)
    if p < warmup_ratio:
        return alpha * (p / max(warmup_ratio, 1e-6))
    return alpha


def run_color_stage(
    pipe: StableDiffusionPipeline,
    img01_hwc: torch.Tensor,
    pal01_nc3: torch.Tensor,
    *,
    steps: int = 24,
    denoise: float = 0.30,
    temp: float = 24.0,
    alpha: float = 0.04,
    warmup: float = 0.25,
    stop_threshold: float | None = None,
    patience: int = 3,
) -> torch.Tensor:
    """Stage A: palette fitting without tile constraint."""
    device = next(pipe.unet.parameters()).device
    base = _encode_latents(pipe, img01_hwc, device)
    latents, start = _img2img_init(pipe, base, steps, denoise)

    loss_hist: list[float] = []
    timesteps = pipe.scheduler.timesteps[start:]

    for i, t in enumerate(timesteps):
        lat_in = pipe.scheduler.scale_model_input(latents, t)
        noise = pipe.unet(lat_in, t, encoder_hidden_states=None).sample
        lat_next = pipe.scheduler.step(noise, t, latents).prev_sample

        w = _warmup_weight(alpha, i, len(timesteps), warmup)
        lat_next = lat_next.detach().requires_grad_(True)
        img01 = _decode_latents(pipe, lat_next)

        loss = palette_loss_lab(img01, pal01_nc3, temp)
        grad = torch.autograd.grad(loss, lat_next)[0]
        latents = (lat_next - w * grad).detach()

        lv = float(loss.detach().cpu())
        loss_hist.append(lv)
        if stop_threshold is not None and lv < stop_threshold:
            break
        if len(loss_hist) > patience:
            if abs(loss_hist[-1] - loss_hist[-1 - patience]) < 1e-4:
                break

    out = _decode_latents(pipe, latents)[0].permute(1, 2, 0)  # (H,W,3)
    return out


def run_pixel_stage(
    pipe: StableDiffusionPipeline,
    img01_hwc: torch.Tensor,
    pal01_nc3: torch.Tensor,
    *,
    steps: int = 24,
    denoise: float = 0.25,
    temp: float = 24.0,
    alpha: float = 0.06,
    warmup: float = 0.25,
    tile: int = 8,
    k: int = 2,
    anchor_img01_hwc: torch.Tensor | None = None,
    anchor_weight: float = 0.5,
) -> torch.Tensor:
    """Stage B: pixel-ish tile proxy optimization."""
    device = next(pipe.unet.parameters()).device
    base = _encode_latents(pipe, img01_hwc, device)
    latents, start = _img2img_init(pipe, base, steps, denoise)
    timesteps = pipe.scheduler.timesteps[start:]

    if anchor_img01_hwc is not None:
        anchor = anchor_img01_hwc.permute(2, 0, 1).unsqueeze(0).to(device)
        anchor_lab = rgb01_to_lab(anchor.permute(0, 2, 3, 1))
    else:
        anchor_lab = None

    for i, t in enumerate(timesteps):
        lat_in = pipe.scheduler.scale_model_input(latents, t)
        noise = pipe.unet(lat_in, t, encoder_hidden_states=None).sample
        lat_next = pipe.scheduler.step(noise, t, latents).prev_sample

        w = _warmup_weight(alpha, i, len(timesteps), warmup)
        lat_next = lat_next.detach().requires_grad_(True)
        img01 = _decode_latents(pipe, lat_next)

        loss = tile_proxy_loss(img01, pal01_nc3, temp, tile=tile, k=k)
        if anchor_lab is not None:
            img_lab = rgb01_to_lab(img01.permute(0, 2, 3, 1))
            loss = loss + anchor_weight * torch.nn.functional.mse_loss(img_lab, anchor_lab)

        grad = torch.autograd.grad(loss, lat_next)[0]
        latents = (lat_next - w * grad).detach()

    out = _decode_latents(pipe, latents)[0].permute(1, 2, 0)
    return out


def run_full(
    pipe: StableDiffusionPipeline,
    img01_hwc: torch.Tensor,
    pal01_nc3: torch.Tensor,
    *,
    color_cfg: dict | None = None,
    pixel_cfg: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run color then pixel stages."""
    color_cfg = color_cfg or {}
    pixel_cfg = pixel_cfg or {}

    a = run_color_stage(pipe, img01_hwc, pal01_nc3, **color_cfg)
    b = run_pixel_stage(pipe, a, pal01_nc3, anchor_img01_hwc=a, **pixel_cfg)
    return a, b


def sweep_variants(
    pipe: StableDiffusionPipeline,
    img01_hwc: torch.Tensor,
    pal01_nc3: torch.Tensor,
    *,
    variants: int = 8,
    base_alpha: float = 0.06,
    base_denoise: float = 0.35,
    steps: int = 24,
    temp: float = 24.0,
):
    """Sweep alpha/denoise to generate multiple outputs."""
    outs = []
    for i in range(variants):
        a = float(base_alpha * (0.5 + i / (max(1, variants - 1)) * 1.0))
        d = float(base_denoise * (0.8 + i / (max(1, variants - 1)) * 0.4))
        out = run_color_stage(
            pipe,
            img01_hwc,
            pal01_nc3,
            steps=steps,
            denoise=d,
            temp=temp,
            alpha=a,
        )
        outs.append(out)
    return outs


__all__ = [
    "run_color_stage",
    "run_pixel_stage",
    "run_full",
    "sweep_variants",
]
