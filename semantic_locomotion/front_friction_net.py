"""CNN from a front-camera RGB frame to the pre-contact material vector.

The locomotion policy stays the official MLP. This network is the only place
the image is read. Its output is (friction coefficient, scaled log stiffness),
which is concatenated onto the policy observation.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

CHECKPOINT_NAME = "front_friction.pt"
MU_MAE_GATE = 0.08
DEFAULT_GROUND_CROP = 0.55
MU_ROCK_OUT = 0.80
MU_WET_OUT = 0.15
LOGK_ROCK_OUT = math.log10(2.0e5) / 5.0
LOGK_WET_OUT = math.log10(2.0e3) / 5.0


def checkpoint_path() -> Path:
    return Path(__file__).resolve().parent.parent / "checkpoints" / CHECKPOINT_NAME


def preprocess_rgb(rgb: torch.Tensor, *, ground_crop: float = DEFAULT_GROUND_CROP) -> torch.Tensor:
    """Normalize RGB and keep the lower ``ground_crop`` fraction (ground-only view)."""
    if rgb.dtype == torch.uint8:
        image = rgb.float() / 255.0
    else:
        image = rgb.float()
        if image.numel() > 0 and float(image.detach().max()) > 1.5:
            image = image / 255.0
    if 0.0 < ground_crop < 1.0:
        height = image.shape[1]
        start = int(height * (1.0 - ground_crop))
        image = image[:, start:, :, :]
    return image


class FrontFrictionNet(nn.Module):
    def __init__(self, *, ground_crop: float = DEFAULT_GROUND_CROP) -> None:
        super().__init__()
        self.ground_crop = ground_crop
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 2)

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        image = preprocess_rgb(rgb, ground_crop=self.ground_crop)
        return self.conv(image.permute(0, 3, 1, 2).contiguous()).flatten(1)

    def logits(self, rgb: torch.Tensor) -> torch.Tensor:
        """Class logits: index 0 = rock, 1 = wet."""
        return self.head(self.encode(rgb))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: (N, H, W, 3). Returns (mu, log10(K)/5) via a 2-way softmax."""
        prob = torch.softmax(self.logits(rgb), dim=-1)
        mu = prob[:, 0] * MU_ROCK_OUT + prob[:, 1] * MU_WET_OUT
        log_k = prob[:, 0] * LOGK_ROCK_OUT + prob[:, 1] * LOGK_WET_OUT
        return torch.stack((mu, log_k), dim=-1)


def save_net(
    net: FrontFrictionNet,
    path: Path | None = None,
    *,
    val_mu_mae: float | None = None,
    max_mu_mae: float = MU_MAE_GATE,
) -> Path:
    """Save ``net`` only when the validation friction MAE gate passes.

    If ``val_mu_mae`` is provided and exceeds ``max_mu_mae``, any stale checkpoint
    at ``path`` is deleted and ``ValueError`` is raised so downstream locomotion
    training cannot accidentally load a failed or stale file.
    """
    path = checkpoint_path() if path is None else Path(path)
    if val_mu_mae is not None and (not torch.isfinite(torch.tensor(val_mu_mae)) or val_mu_mae > max_mu_mae):
        if path.is_file():
            path.unlink()
        raise ValueError(
            f"Refusing to save {path}: val_mu_mae={val_mu_mae:.4f} exceeds gate {max_mu_mae:.4f}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "state_dict": net.state_dict(),
        "gate_passed": True if val_mu_mae is None else bool(val_mu_mae <= max_mu_mae),
        "val_mu_mae": float(0.0 if val_mu_mae is None else val_mu_mae),
        "max_mu_mae": float(max_mu_mae),
        "ground_crop": float(getattr(net, "ground_crop", DEFAULT_GROUND_CROP)),
    }
    torch.save(payload, path)
    return path


def load_net(
    path: Path | None = None,
    device: torch.device | str = "cpu",
    *,
    max_mu_mae: float = MU_MAE_GATE,
) -> FrontFrictionNet:
    path = checkpoint_path() if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Train the front camera first with scripts/train_front_friction.py "
            "on the Isaac Lab machine. The policy does not invent friction without that file."
        )
    payload = torch.load(path, map_location=device, weights_only=True)
    if not payload.get("gate_passed", False):
        raise RuntimeError(f"Refusing {path}: checkpoint did not pass the friction MAE gate.")
    val_mu_mae = payload.get("val_mu_mae", None)
    if val_mu_mae is not None and float(val_mu_mae) > max_mu_mae:
        raise RuntimeError(
            f"Refusing {path}: recorded val_mu_mae={float(val_mu_mae):.4f} exceeds {max_mu_mae:.4f}."
        )
    crop = float(payload.get("ground_crop", DEFAULT_GROUND_CROP))
    net = FrontFrictionNet(ground_crop=crop)
    net.load_state_dict(payload["state_dict"])
    net.to(device)
    net.eval()
    return net
