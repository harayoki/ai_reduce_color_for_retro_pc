"""Evaluation rules: Lab conversion, palette fitting, and tile proxy losses."""
import torch


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def _linear_rgb_to_xyz(rgb_lin: torch.Tensor) -> torch.Tensor:
    M = torch.tensor(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        device=rgb_lin.device, dtype=rgb_lin.dtype
    )
    return rgb_lin @ M.T


def _xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    white = torch.tensor([0.95047, 1.00000, 1.08883], device=xyz.device, dtype=xyz.dtype)
    x = xyz / white

    eps = 216 / 24389  # (6/29)^3
    k = 24389 / 27

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > eps, t ** (1 / 3), (k * t + 16) / 116)

    fx, fy, fz = f(x[..., 0]), f(x[..., 1]), f(x[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=-1)


def rgb01_to_lab(rgb01: torch.Tensor) -> torch.Tensor:
    """Convert RGB in [0, 1] to Lab."""
    return _xyz_to_lab(_linear_rgb_to_xyz(_srgb_to_linear(rgb01)))


def soft_palette_assign_lab(img01_bchw: torch.Tensor, pal01_nc3: torch.Tensor, temp: float):
    """
    Convert image to Lab and perform soft assignment to palette colors.
    Returns reconstructed Lab image and weights.
    """
    img_lab = rgb01_to_lab(img01_bchw.permute(0, 2, 3, 1))  # (B, H, W, 3)
    pal_lab = rgb01_to_lab(pal01_nc3)                       # (N, 3)

    B, H, W, _ = img_lab.shape
    x = img_lab.reshape(B, H * W, 3)
    d2 = ((x[:, :, None, :] - pal_lab[None, None, :, :]) ** 2).sum(-1)  # (B, HW, N)
    w = torch.softmax(-temp * d2, dim=-1)                              # (B, HW, N)
    recon = w @ pal_lab                                               # (B, HW, 3)

    return recon.reshape(B, H, W, 3), w.reshape(B, H, W, -1)


def palette_loss_lab(img01_bchw: torch.Tensor, pal01_nc3: torch.Tensor, temp: float) -> torch.Tensor:
    """Palette reconstruction loss in Lab space."""
    recon, _ = soft_palette_assign_lab(img01_bchw, pal01_nc3, temp)
    img_lab = rgb01_to_lab(img01_bchw.permute(0, 2, 3, 1))
    return torch.nn.functional.mse_loss(recon, img_lab)


def tile_proxy_loss_from_weights(w_bhwn: torch.Tensor, tile: int, k: int) -> torch.Tensor:
    """Tile-level proxy loss combining entropy and spill outside top-k colors."""
    B, H, W, N = w_bhwn.shape
    assert H % tile == 0 and W % tile == 0

    wt = w_bhwn.view(B, H // tile, tile, W // tile, tile, N).sum(dim=(2, 4))  # (B, HT, WT, N)
    p = wt / (wt.sum(dim=-1, keepdim=True) + 1e-8)

    ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean()
    topk = torch.topk(p, k=k, dim=-1).values.sum(dim=-1)
    spill = (1.0 - topk).mean()
    return ent + 2.0 * spill


def tile_proxy_loss(img01_bchw: torch.Tensor, pal01_nc3: torch.Tensor, temp: float, tile: int = 8, k: int = 2) -> torch.Tensor:
    """Soft-assign to palette then evaluate tile proxy loss."""
    _, w = soft_palette_assign_lab(img01_bchw, pal01_nc3, temp)
    return tile_proxy_loss_from_weights(w, tile=tile, k=k)
