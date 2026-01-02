# =========================
# src/msxfilter/io.py
# =========================

def load_image_01(path, width, height):
    """
    画像をRGBで読み込み、(H,W,3) float32 [0,1] を返す。
    """
    img = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0
    return torch.from_numpy(arr)  # (H,W,3)

def save_image_01(path, img01_hwc):
    """
    (H,W,3) float [0,1] をPNG等で保存。
    """
    arr = (img01_hwc.clamp(0,1).cpu().numpy() * 255.0).round().astype("uint8")
    Image.fromarray(arr, mode="RGB").save(path)

def load_palette_txt(path):
    """
    txt: 'R G B' または 'R,G,B' を1行1色で読み込み、(N,3) float [0,1]。
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) != 3:
            raise ValueError(f"bad palette line: {line}")
        r, g, b = [float(x) for x in parts]
        rows.append([r, g, b])
    if not rows:
        raise ValueError("palette empty")
    return torch.tensor(rows, dtype=torch.float32) / 255.0

  
# =========================
# src/msxfilter/rules.py
# 大事なところだけ：Lab変換、パレット寄せ、tile proxy
# =========================

def _srgb_to_linear(x):
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def _linear_rgb_to_xyz(rgb_lin):
    M = torch.tensor(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        device=rgb_lin.device, dtype=rgb_lin.dtype
    )
    return rgb_lin @ M.T

def _xyz_to_lab(xyz):
    white = torch.tensor([0.95047, 1.00000, 1.08883], device=xyz.device, dtype=xyz.dtype)
    x = xyz / white

    eps = 216 / 24389  # (6/29)^3
    k = 24389 / 27

    def f(t):
        return torch.where(t > eps, t ** (1/3), (k * t + 16) / 116)

    fx, fy, fz = f(x[..., 0]), f(x[..., 1]), f(x[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=-1)

def rgb01_to_lab(rgb01):
    """
    rgb01: (...,3) in [0,1] → (...,3) Lab
    """
    return _xyz_to_lab(_linear_rgb_to_xyz(_srgb_to_linear(rgb01)))

def soft_palette_assign_lab(img01_bchw, pal01_nc3, temp):
    """
    画像をLabにして、各ピクセルをパレットN色へsoft割当。
    戻り:
      recon_lab: (B,H,W,3)
      w: (B,H,W,N)
    """
    img_lab = rgb01_to_lab(img01_bchw.permute(0,2,3,1))     # (B,H,W,3)
    pal_lab = rgb01_to_lab(pal01_nc3)                        # (N,3)

    B, H, W, _ = img_lab.shape
    x = img_lab.reshape(B, H*W, 3)                           # (B,HW,3)
    d2 = ((x[:, :, None, :] - pal_lab[None, None, :, :])**2).sum(-1)  # (B,HW,N)
    w = torch.softmax(-temp * d2, dim=-1)                    # (B,HW,N)
    recon = w @ pal_lab                                      # (B,HW,3)

    return recon.reshape(B,H,W,3), w.reshape(B,H,W,-1)

def palette_loss_lab(img01_bchw, pal01_nc3, temp):
    """
    パレットに寄せるロス（Lab空間での再構成誤差）。
    """
    recon, _ = soft_palette_assign_lab(img01_bchw, pal01_nc3, temp)
    img_lab = rgb01_to_lab(img01_bchw.permute(0,2,3,1))
    return torch.nn.functional.mse_loss(recon, img_lab)

def tile_proxy_loss_from_weights(w_bhwn, tile, k):
    """
    w: (B,H,W,N) ＝パレットsoft割当確率
    tileごとにN次元分布を集計し、上位k以外の質量(spill)とエントロピーを罰する。
    """
    B, H, W, N = w_bhwn.shape
    assert H % tile == 0 and W % tile == 0

    wt = w_bhwn.view(B, H//tile, tile, W//tile, tile, N).sum(dim=(2,4))  # (B,HT,WT,N)
    p = wt / (wt.sum(dim=-1, keepdim=True) + 1e-8)

    ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean()
    topk = torch.topk(p, k=k, dim=-1).values.sum(dim=-1)
    spill = (1.0 - topk).mean()
    return ent + 2.0 * spill

def tile_proxy_loss(img01_bchw, pal01_nc3, temp, tile=8, k=2):
    """
    画像→soft割当→tile proxy
    """
    _, w = soft_palette_assign_lab(img01_bchw, pal01_nc3, temp)
    return tile_proxy_loss_from_weights(w, tile=tile, k=k)

# =========================
# src/msxfilter/pipeline.py
# 2段処理：A(色味)→B(ドット寄せ)
# “AI提案”はSDを使う想定（prompt無し・img2img）
# =========================

def _encode_latents(pipe, img01_hwc, device):
    t = img01_hwc.permute(2,0,1).unsqueeze(0).to(device)  # (1,3,H,W)
    t = t * 2 - 1
    with torch.no_grad():
        z = pipe.vae.encode(t).latent_dist.sample()
        z = z * pipe.vae.config.scaling_factor
    return z

def _decode_latents(pipe, latents):
    z = latents / pipe.vae.config.scaling_factor
    img = pipe.vae.decode(z).sample
    return (img / 2 + 0.5).clamp(0,1)  # (1,3,H,W)

def _img2img_init(pipe, base_latents, steps, denoise):
    pipe.scheduler.set_timesteps(steps, device=base_latents.device)
    start = int(steps * denoise)
    start = max(0, min(steps-1, start))
    t0 = pipe.scheduler.timesteps[start]
    return pipe.scheduler.add_noise(base_latents, torch.randn_like(base_latents), t0), start

def _warmup_weight(alpha, i, total, warmup_ratio):
    if total <= 1:
        return alpha
    p = i / (total - 1)
    if p < warmup_ratio:
        return alpha * (p / max(warmup_ratio, 1e-6))
    return alpha

def run_color_stage(pipe, img01_hwc, pal01_nc3, *,
                    steps=24, denoise=0.30, temp=24.0,
                    alpha=0.04, warmup=0.25,
                    stop_threshold=None, patience=3):
    """
    A: 色味変換（92色などへ寄せる）
    - tile制約は入れない（色逃げの確定を優先）
    """
    device = next(pipe.unet.parameters()).device
    base = _encode_latents(pipe, img01_hwc, device)
    latents, start = _img2img_init(pipe, base, steps, denoise)

    loss_hist = []
    timesteps = pipe.scheduler.timesteps[start:]

    for i, t in enumerate(timesteps):
        # SDの提案（なめらか復元）
        lat_in = pipe.scheduler.scale_model_input(latents, t)
        noise = pipe.unet(lat_in, t, encoder_hidden_states=None).sample
        lat_next = pipe.scheduler.step(noise, t, latents).prev_sample

        # ルールでフィードバック（パレット寄せのみ）
        w = _warmup_weight(alpha, i, len(timesteps), warmup)
        lat_next = lat_next.detach().requires_grad_(True)
        img01 = _decode_latents(pipe, lat_next)

        loss = palette_loss_lab(img01, pal01_nc3, temp)
        grad = torch.autograd.grad(loss, lat_next)[0]
        latents = (lat_next - w * grad).detach()

        # 早期停止（任意）
        lv = float(loss.detach().cpu())
        loss_hist.append(lv)
        if stop_threshold is not None and lv < stop_threshold:
            break
        if len(loss_hist) > patience:
            if abs(loss_hist[-1] - loss_hist[-1-patience]) < 1e-4:
                break

    out = _decode_latents(pipe, latents)[0].permute(1,2,0)  # (H,W,3)
    return out

def run_pixel_stage(pipe, img01_hwc, pal01_nc3, *,
                    steps=24, denoise=0.25, temp=24.0,
                    alpha=0.06, warmup=0.25,
                    tile=8, k=2,
                    anchor_img01_hwc=None, anchor_weight=0.5):
    """
    B: ドット寄せ（tile proxy）
    - anchorを渡すと「色逃げ（A結果）」を壊しにくくなる
    """
    device = next(pipe.unet.parameters()).device
    base = _encode_latents(pipe, img01_hwc, device)
    latents, start = _img2img_init(pipe, base, steps, denoise)
    timesteps = pipe.scheduler.timesteps[start:]

    # anchor（固定目標）をLabで近づける罰則にする場合
    if anchor_img01_hwc is not None:
        anchor = anchor_img01_hwc.permute(2,0,1).unsqueeze(0).to(device)  # (1,3,H,W)
        anchor_lab = rgb01_to_lab(anchor.permute(0,2,3,1))                # (1,H,W,3)
    else:
        anchor_lab = None

    for i, t in enumerate(timesteps):
        lat_in = pipe.scheduler.scale_model_input(latents, t)
        noise = pipe.unet(lat_in, t, encoder_hidden_states=None).sample
        lat_next = pipe.scheduler.step(noise, t, latents).prev_sample

        w = _warmup_weight(alpha, i, len(timesteps), warmup)
        lat_next = lat_next.detach().requires_grad_(True)
        img01 = _decode_latents(pipe, lat_next)

        # tile proxy（空間制約寄せ）
        loss = tile_proxy_loss(img01, pal01_nc3, temp, tile=tile, k=k)

        # 色逃げ保持（任意）：A結果から大きく離れない
        if anchor_lab is not None:
            img_lab = rgb01_to_lab(img01.permute(0,2,3,1))
            loss = loss + anchor_weight * torch.nn.functional.mse_loss(img_lab, anchor_lab)

        grad = torch.autograd.grad(loss, lat_next)[0]
        latents = (lat_next - w * grad).detach()

    out = _decode_latents(pipe, latents)[0].permute(1,2,0)
    return out

def run_full(pipe, img01_hwc, pal01_nc3, *,
             color_cfg=None, pixel_cfg=None):
    """
    A→Bをまとめて実行
    """
    color_cfg = color_cfg or {}
    pixel_cfg = pixel_cfg or {}

    a = run_color_stage(pipe, img01_hwc, pal01_nc3, **color_cfg)
    b = run_pixel_stage(pipe, a, pal01_nc3, anchor_img01_hwc=a, **pixel_cfg)
    return a, b

def sweep_variants(pipe, img01_hwc, pal01_nc3, *,
                   variants=8, base_alpha=0.06, base_denoise=0.35, steps=24, temp=24.0):
    """
    8枚くらい出す用。alpha/denoiseを線形に振るだけの最小実装。
    """
    outs = []
    for i in range(variants):
        a = float(base_alpha * (0.5 + i/(max(1,variants-1)) * 1.0))   # 0.5〜1.5倍
        d = float(base_denoise * (0.8 + i/(max(1,variants-1)) * 0.4)) # 0.8〜1.2倍
        out = run_color_stage(pipe, img01_hwc, pal01_nc3,
                              steps=steps, denoise=d, temp=temp, alpha=a)
        outs.append(out)
    return outs

# =========================
# src/msxfilter/cli.py
# 大事なところだけ：A/B/full を呼ぶ
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["color", "pixel", "full", "sweep"], required=True)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", type=Path)          # color/pixel用
    ap.add_argument("--output_dir", type=Path)      # full/sweep用
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
            pipe, img01, pal01,
            color_cfg=dict(steps=args.steps, denoise=args.denoise, alpha=args.alpha, temp=args.temp),
            pixel_cfg=dict(steps=args.steps, denoise=max(0.0, args.denoise-0.05), alpha=args.alpha, temp=args.temp, k=args.tile_k),
        )
        save_image_01(args.output_dir / "A_color.png", a)
        save_image_01(args.output_dir / "B_pixelish.png", b)

    elif args.mode == "sweep":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        outs = sweep_variants(
            pipe, img01, pal01,
            variants=args.variants, base_alpha=args.alpha, base_denoise=args.denoise,
            steps=args.steps, temp=args.temp,
        )
        for i, o in enumerate(outs):
            save_image_01(args.output_dir / f"out_{i:02d}.png", o)

if __name__ == "__main__":
    main()

  
