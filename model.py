import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def squash(inputs: torch.Tensor, axis: int = -1, eps: float = 1e-9) -> torch.Tensor:
    """
    Capsule squash nonlinearity (routing-by-agreement).
    """
    norm_sq = torch.sum(inputs * inputs, dim=axis, keepdim=True)
    scale = norm_sq / (1.0 + norm_sq)
    return scale * inputs / torch.sqrt(norm_sq + eps)


class CapsuleLayer(nn.Module):
    """
    Fully connected capsule routing layer.

    input : [B, N_in, D_in]
    output: [B, N_out, D_out]
    """

    def __init__(
        self,
        num_capsules: int,
        num_route_nodes: int,
        in_channels: int,
        out_channels: int,
        num_iterations: int = 3,
    ):
        super().__init__()
        self.num_capsules = num_capsules
        self.num_route_nodes = num_route_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_iterations = num_iterations

        # [N_out, N_in, D_out, D_in]
        self.W = nn.Parameter(
            0.01 * torch.randn(num_capsules, num_route_nodes, out_channels, in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_in, D_in]
        bsz = x.size(0)
        if x.size(1) != self.num_route_nodes:
            raise ValueError(f"Expected N_in={self.num_route_nodes}, got {x.size(1)}")
        if x.size(2) != self.in_channels:
            raise ValueError(f"Expected D_in={self.in_channels}, got {x.size(2)}")

        # u_hat: [B, N_out, N_in, D_out]
        u_hat = torch.einsum("oidc,bic->boid", self.W, x)

        # Routing logits: [B, N_out, N_in]
        b_ij = torch.zeros(
            bsz, self.num_capsules, self.num_route_nodes, device=x.device, dtype=x.dtype
        )

        for i in range(self.num_iterations):
            # Coupling coefficients across output capsules
            c_ij = F.softmax(b_ij, dim=1)
            # Weighted sum over input capsules
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=2)  # [B, N_out, D_out]
            v_j = squash(s_j, axis=-1)

            if i < self.num_iterations - 1:
                # Agreement: [B, N_out, N_in]
                agreement = torch.einsum("boid,bod->boi", u_hat, v_j)
                b_ij = b_ij + agreement

        return v_j


class LBPBlock(nn.Module):
    """
    Local binary pattern-style maps (8-neighbor comparator) per channel.
    """

    def __init__(self):
        super().__init__()
        self.offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        _, _, h, w = x.shape
        xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
        center = xp[:, :, 1 : 1 + h, 1 : 1 + w]
        outs = []
        for dy, dx in self.offsets:
            yy = 1 + dy
            xx = 1 + dx
            nbr = xp[:, :, yy : yy + h, xx : xx + w]
            outs.append((nbr >= center).to(x.dtype))
        return torch.cat(outs, dim=1)  # [B, 8C, H, W]


class ConvPrimaryCaps(nn.Module):
    """
    Primary capsule projection from fused feature map.
    """

    def __init__(self, in_channels: int, num_capsules: int = 8, capsule_dim: int = 8):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.proj = nn.Conv2d(
            in_channels,
            num_capsules * capsule_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        x = self.proj(x)  # [B, num_caps*caps_dim, H, W]
        b, _, h, w = x.shape
        x = x.view(b, self.num_capsules, self.capsule_dim, h, w)
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # [B,H,W,Ncaps,D]
        x = x.view(b, h * w * self.num_capsules, self.capsule_dim)
        x = squash(x, axis=-1)
        return x


class TimmHRNetBackbone(nn.Module):
    """
    HRNet backbone using external timm implementation.

    - Uses timm feature extraction API (`features_only=True`)
    - Removes classification head behavior by consuming intermediate features
    - Produces fixed feature map [B, 64, 56, 56] for capsule fusion
    """

    def __init__(self, model_name: str = "hrnet_w18", pretrained: bool = False):
        super().__init__()
        self.model_name = model_name

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        feat_info = self.backbone.feature_info.get_dicts()
        self.feature_channels = [d["num_chs"] for d in feat_info]

        # Aggregate all selected feature maps into one tensor and project to 64 channels
        in_ch = sum(self.feature_channels)
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm HRNet can hit branch-fusion mismatches on non-aligned spatial sizes (e.g., 299x299).
        # Pad to the next multiple of 32 before backbone forward, then keep downstream fixed to 56x56.
        h, w = x.shape[-2:]
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32
        if pad_h != 0 or pad_w != 0:
            x_safe = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_safe = x

        feats = self.backbone(x_safe)  # list of feature maps low->high semantic

        # Choose highest spatial resolution among outputs as fusion target
        target_h, target_w = feats[0].shape[-2:]
        resized = [
            F.interpolate(f, size=(target_h, target_w), mode="nearest")
            if f.shape[-2:] != (target_h, target_w)
            else f
            for f in feats
        ]
        y = torch.cat(resized, dim=1)
        y = self.fuse_proj(y)
        y = F.interpolate(y, size=(56, 56), mode="bilinear", align_corners=False)
        return y  # [B,64,56,56]


class HRNetLBPCapsNet(nn.Module):
    """
    Paper-aligned architecture (external HRNet variant):
      1) timm HRNet high-resolution backbone (no classifier head path)
      2) LBP texture extraction in HSV + YCbCr spaces
      3) Feature fusion: HRNet(64x56x56) + texture(6x56x56)
      4) Capsule classification via routing-by-agreement

    Output:
      class capsule lengths [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 2,
        primary_capsules: int = 8,
        primary_dim: int = 8,
        digit_dim: int = 16,
        routing_iters: int = 3,
        hrnet_name: str = "hrnet_w18",
        hrnet_pretrained: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        # External HRNet backbone
        self.hrnet = TimmHRNetBackbone(
            model_name=hrnet_name, pretrained=hrnet_pretrained
        )
        self.lbp = LBPBlock()

        # LBP over 3 channels => 24 maps per space; combine HSV + YCbCr -> averaged 24 maps
        self.texture_reduce = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 6, kernel_size=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
        )

        # Fuse HRNet + texture
        self.fusion = nn.Sequential(
            nn.Conv2d(64 + 6, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Capsule stack
        self.primary_caps = ConvPrimaryCaps(
            in_channels=64,
            num_capsules=primary_capsules,
            capsule_dim=primary_dim,
        )

        # route_nodes = 56 * 56 * primary_capsules
        route_nodes = 56 * 56 * primary_capsules
        self.digit_caps = CapsuleLayer(
            num_capsules=num_classes,
            num_route_nodes=route_nodes,
            in_channels=primary_dim,
            out_channels=digit_dim,
            num_iterations=routing_iters,
        )

    @staticmethod
    def _rgb_to_hsv(x: torch.Tensor) -> torch.Tensor:
        # x in [0,1], [B,3,H,W]
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        maxc, _ = torch.max(x, dim=1, keepdim=True)
        minc, _ = torch.min(x, dim=1, keepdim=True)
        delta = maxc - minc + 1e-6

        v = maxc
        s = delta / (maxc + 1e-6)

        h = torch.zeros_like(v)
        mask_r = maxc == r
        mask_g = maxc == g
        mask_b = maxc == b

        h = torch.where(mask_r, ((g - b) / delta) % 6.0, h)
        h = torch.where(mask_g, ((b - r) / delta) + 2.0, h)
        h = torch.where(mask_b, ((r - g) / delta) + 4.0, h)
        h = h / 6.0

        return torch.cat([h, s, v], dim=1)

    @staticmethod
    def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
        # x in [0,1], [B,3,H,W]
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 0.564 * (b - y) + 0.5
        cr = 0.713 * (r - y) + 0.5
        return torch.cat([y, cb, cr], dim=1)

    def _texture_features(self, x: torch.Tensor) -> torch.Tensor:
        x01 = torch.clamp(x, 0.0, 1.0)

        hsv = self._rgb_to_hsv(x01)
        ycbcr = self._rgb_to_ycbcr(x01)

        lbp_hsv = self.lbp(hsv)  # [B,24,H,W]
        lbp_ycc = self.lbp(ycbcr)  # [B,24,H,W]

        lbp_hsv = F.interpolate(lbp_hsv, size=(56, 56), mode="nearest")
        lbp_ycc = F.interpolate(lbp_ycc, size=(56, 56), mode="nearest")

        lbp = 0.5 * (lbp_hsv + lbp_ycc)
        tex = self.texture_reduce(lbp)  # [B,6,56,56]
        return tex

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hr = self.hrnet(x)  # [B,64,56,56]
        tex = self._texture_features(x)  # [B,6,56,56]

        feat = self.fusion(torch.cat([hr, tex], dim=1))  # [B,64,56,56]

        pri = self.primary_caps(feat)  # [B,N_in,8]
        dig = self.digit_caps(pri)  # [B,num_classes,16]

        lengths = torch.norm(dig, dim=-1)  # [B,num_classes]
        return lengths


# Backward-compatible alias used elsewhere in project
SegCapsCNN = HRNetLBPCapsNet


if __name__ == "__main__":
    model = HRNetLBPCapsNet(num_classes=2, hrnet_name="hrnet_w18", hrnet_pretrained=False)
    dummy = torch.rand(2, 3, 299, 299)
    out = model(dummy)
    print("Output shape:", out.shape)  # expected: [2, 2]