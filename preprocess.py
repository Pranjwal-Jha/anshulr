from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


TensorLike = Union[torch.Tensor, np.ndarray]


def normalize_data(q: TensorLike, eps: float = 1e-12) -> TensorLike:
    """
    Signal normalization from the paper (Eq. 1):
        q_norm = (q - q_min) / (q_max - q_min)

    Works with torch.Tensor or np.ndarray.
    """
    if isinstance(q, torch.Tensor):
        q_min = torch.min(q)
        q_max = torch.max(q)
        denom = (q_max - q_min).clamp_min(eps)
        return (q - q_min) / denom

    q_arr = np.asarray(q)
    q_min = np.min(q_arr)
    q_max = np.max(q_arr)
    denom = max(float(q_max - q_min), eps)
    return (q_arr - q_min) / denom


@dataclass
class AugmentationConfig:
    random_crop_size: Tuple[int, int] = (299, 299)
    resize_size: Tuple[int, int] = (320, 320)  # resize first, then random crop to 299x299
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    max_rotation_deg: float = 15.0
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.05


class DeepfakePreprocessor:
    """
    Preprocessing aligned to the paper:
      1) Spatial normalization with Lanczos resize
      2) Random crop to 299x299
      3) Data augmentation (rotation, flips, color changes)
      4) Signal min-max normalization
      5) Optional 2D slicing (XZ / YX) from 3D video tensor
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (299, 299),
        cfg: AugmentationConfig | None = None,
    ) -> None:
        self.target_size = target_size
        self.cfg = cfg or AugmentationConfig(random_crop_size=target_size)

        # Training pipeline (paper-style augmentation + Lanczos + crop)
        self.train_transform = T.Compose(
            [
                T.Resize(self.cfg.resize_size, interpolation=InterpolationMode.LANCZOS),
                T.RandomCrop(self.cfg.random_crop_size),
                T.RandomHorizontalFlip(p=self.cfg.hflip_p),
                T.RandomVerticalFlip(p=self.cfg.vflip_p),
                T.RandomRotation(degrees=self.cfg.max_rotation_deg, interpolation=InterpolationMode.BILINEAR),
                T.ColorJitter(
                    brightness=self.cfg.color_jitter_brightness,
                    contrast=self.cfg.color_jitter_contrast,
                    saturation=self.cfg.color_jitter_saturation,
                    hue=self.cfg.color_jitter_hue,
                ),
                T.ToTensor(),  # [0,1], CxHxW
            ]
        )

        # Eval/inference pipeline (deterministic)
        self.eval_transform = T.Compose(
            [
                T.Resize(self.target_size, interpolation=InterpolationMode.LANCZOS),
                T.ToTensor(),
            ]
        )

    # ------------------------
    # Core image preprocessing
    # ------------------------
    def preprocess_image(self, img: Image.Image, train: bool = True) -> torch.Tensor:
        """
        Preprocess a single PIL image.
        Returns tensor [3, H, W], min-max normalized as per paper.
        """
        transform = self.train_transform if train else self.eval_transform
        x = transform(img)
        x = normalize_data(x)
        return x

    def preprocess_frame_tensor(self, frame: torch.Tensor, train: bool = True) -> torch.Tensor:
        """
        Preprocess a single frame given as torch tensor:
          - accepts [H,W,C] or [C,H,W]
          - values can be uint8 [0,255] or float
        """
        if frame.ndim != 3:
            raise ValueError(f"Expected 3D frame tensor, got shape {tuple(frame.shape)}")

        if frame.shape[0] in (1, 3):  # [C,H,W]
            chw = frame
        elif frame.shape[-1] in (1, 3):  # [H,W,C]
            chw = frame.permute(2, 0, 1)
        else:
            raise ValueError("Frame must have channel dimension of size 1 or 3.")

        if chw.dtype != torch.float32:
            chw = chw.float()

        if torch.max(chw) > 1.0:
            chw = chw / 255.0

        # to PIL for torchvision augment ops
        pil_img = TF.to_pil_image(chw.clamp(0.0, 1.0))
        return self.preprocess_image(pil_img, train=train)

    # ------------------------
    # Video-level preprocessing
    # ------------------------
    def preprocess_video_frames(
        self,
        frames: Sequence[Union[Image.Image, torch.Tensor]],
        train: bool = True,
    ) -> torch.Tensor:
        """
        Preprocess sequence of frames.
        Returns tensor [T, 3, 299, 299].
        """
        processed: List[torch.Tensor] = []
        for fr in frames:
            if isinstance(fr, Image.Image):
                x = self.preprocess_image(fr, train=train)
            elif isinstance(fr, torch.Tensor):
                x = self.preprocess_frame_tensor(fr, train=train)
            else:
                raise TypeError(f"Unsupported frame type: {type(fr)}")
            processed.append(x)

        if not processed:
            raise ValueError("frames is empty.")
        return torch.stack(processed, dim=0)

    # ------------------------
    # 2D slicing from 3D volume
    # ------------------------
    @staticmethod
    def _to_video_tensor(video_frames: TensorLike) -> torch.Tensor:
        """
        Convert input to canonical [T, C, H, W] tensor.
        Accepted:
          - torch.Tensor [T,C,H,W] or [T,H,W,C]
          - np.ndarray with same forms
          - list/tuple of frame tensors or arrays
        """
        if isinstance(video_frames, (list, tuple)):
            # try stacking list of frames
            elems = []
            for x in video_frames:
                if isinstance(x, np.ndarray):
                    t = torch.from_numpy(x)
                elif isinstance(x, torch.Tensor):
                    t = x
                else:
                    raise TypeError(f"Unsupported element type in frames list: {type(x)}")
                elems.append(t)
            video = torch.stack(elems, dim=0)
        elif isinstance(video_frames, np.ndarray):
            video = torch.from_numpy(video_frames)
        elif isinstance(video_frames, torch.Tensor):
            video = video_frames
        else:
            raise TypeError(f"Unsupported video_frames type: {type(video_frames)}")

        if video.ndim != 4:
            raise ValueError(f"Expected 4D video tensor, got shape {tuple(video.shape)}")

        # [T,H,W,C] -> [T,C,H,W]
        if video.shape[-1] in (1, 3) and video.shape[1] not in (1, 3):
            video = video.permute(0, 3, 1, 2)
        elif video.shape[1] in (1, 3):
            pass
        else:
            raise ValueError("Video must be [T,C,H,W] or [T,H,W,C] with channel size 1 or 3.")

        return video

    def segment_volume(
        self,
        video_frames: TensorLike,
        plane: str = "XZ",
        center_index: int | None = None,
        return_all_channels: bool = True,
    ) -> torch.Tensor:
        """
        Extract 2D slice planes from video volume as discussed in the paper:
          - XZ plane: fix Y (row), vary time + width
          - YX plane: per-frame spatial plane (equivalent to XY)
        Input expected as video tensor [T,C,H,W] (or equivalent accepted by _to_video_tensor).

        Returns:
          - XZ: [C, T, W] (or [T, W] if return_all_channels=False)
          - YX: [T, C, H, W] (or [T, H, W] if return_all_channels=False)
        """
        vol = self._to_video_tensor(video_frames).float()  # [T,C,H,W]
        t, c, h, w = vol.shape

        # normalize signal before slicing, per paper's normalization intent
        vol = normalize_data(vol)

        plane = plane.upper()
        if plane not in {"XZ", "YX", "XY"}:
            raise ValueError("plane must be one of {'XZ', 'YX', 'XY'}")

        if plane == "XZ":
            # choose central Y index if not given
            y_idx = h // 2 if center_index is None else int(center_index)
            y_idx = max(0, min(h - 1, y_idx))
            # [T,C,W] -> [C,T,W]
            xz = vol[:, :, y_idx, :].permute(1, 0, 2).contiguous()
            if return_all_channels:
                return xz
            return xz.mean(dim=0)  # [T,W]

        # YX/XY plane across all frames
        yx = vol  # [T,C,H,W]
        if return_all_channels:
            return yx
        return yx.mean(dim=1)  # [T,H,W]


if __name__ == "__main__":
    # Minimal smoke tests
    pre = DeepfakePreprocessor(target_size=(299, 299))

    # 1) normalize_data
    x = torch.tensor([10.0, 50.0, 100.0])
    print("Normalized:", normalize_data(x))

    # 2) segment_volume
    dummy_video = torch.rand(8, 3, 299, 299)  # [T,C,H,W]
    xz = pre.segment_volume(dummy_video, plane="XZ")
    yx = pre.segment_volume(dummy_video, plane="YX")
    print("XZ shape:", tuple(xz.shape))  # [C,T,W]
    print("YX shape:", tuple(yx.shape))  # [T,C,H,W]