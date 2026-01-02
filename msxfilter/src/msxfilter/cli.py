"""CLI entrypoint for MSX filter pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from .io import load_image_01, load_palette_txt, save_image_01
from .pipeline import run_color_stage, run_full, run_pixel_stage, sweep_variants


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["color", "pixel", "full", "sweep"], required=True)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--output_dir", type=Path)
    ap.add_argument("--palette", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")

    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--denoise", type=float, default=0.35)
    ap.add_argument("--alpha", type=float, default=0.06)
    ap.add_argument("--temp", type=float, default=24.0)
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--tile_k", type=int, default=2)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    img01 = load_image_01(args.input, 256, 192)
    pal01 = load_palette_txt(args.palette).to(device)

    if args.mode == "color":
        out = run_color_stage(pipe, img01, pal01, steps=args.steps, denoise=args.denoise, alpha=args.alpha, temp=args.temp)
        save_image_01(args.output, out)

    elif args.mode == "pixel":
        out = run_pixel_stage(pipe, img01, pal01, steps=args.steps, denoise=args.denoise, alpha=args.alpha, temp=args.temp, k=args.tile_k)
        save_image_01(args.output, out)

    elif args.mode == "full":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        a, b = run_full(
            pipe,
            img01,
            pal01,
            color_cfg=dict(steps=args.steps, denoise=args.denoise, alpha=args.alpha, temp=args.temp),
            pixel_cfg=dict(steps=args.steps, denoise=max(0.0, args.denoise - 0.05), alpha=args.alpha, temp=args.temp, k=args.tile_k),
        )
        save_image_01(args.output_dir / "A_color.png", a)
        save_image_01(args.output_dir / "B_pixelish.png", b)

    elif args.mode == "sweep":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        outs = sweep_variants(
            pipe,
            img01,
            pal01,
            variants=args.variants,
            base_alpha=args.alpha,
            base_denoise=args.denoise,
            steps=args.steps,
            temp=args.temp,
        )
        for i, o in enumerate(outs):
            save_image_01(args.output_dir / f"out_{i:02d}.png", o)


if __name__ == "__main__":
    main()
