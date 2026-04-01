import torch
import torch.nn as nn
import torch.nn.functional as F


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
            raise ValueError(
                f"Expected N_in={self.num_route_nodes}, got {x.size(1)}"
            )
        if x.size(2) != self.in_channels:
            raise ValueError(
                f"Expected D_in={self.in_channels}, got {x.size(2)}"
            )

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


class ConvBNReLU(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BasicBlock(nn.Module):
    """
    Lightweight residual block (2x 3x3 conv).
    """
    expansion = 1

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNReLU(c_in, c_out, k=3, s=stride, p=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out


def _make_branch(c_in: int, c_out: int, num_blocks: int = 2) -> nn.Sequential:
    blocks = [BasicBlock(c_in, c_out, stride=1)]
    for _ in range(num_blocks - 1):
        blocks.append(BasicBlock(c_out, c_out, stride=1))
    return nn.Sequential(*blocks)


class HRNetStage(nn.Module):
    """
    Minimal HRNet-style multi-branch stage with repeated fusion.
    """
    def __init__(self, channels):
        super().__init__()
        # channels list, one per branch
        self.channels = channels
        self.num_branches = len(channels)

        self.branches = nn.ModuleList(
            [_make_branch(ch, ch, num_blocks=2) for ch in channels]
        )

        # fuse layers: each target i receives sum of transformed source j
        self.fuse_layers = nn.ModuleList()
        for i in range(self.num_branches):
            fuse_row = nn.ModuleList()
            for j in range(self.num_branches):
                if i == j:
                    fuse_row.append(nn.Identity())
                elif j > i:
                    # upsample from lower-res j to higher-res i
                    fuse_row.append(
                        nn.Sequential(
                            nn.Conv2d(channels[j], channels[i], kernel_size=1, bias=False),
                            nn.BatchNorm2d(channels[i]),
                        )
                    )
                else:
                    # downsample from higher-res j to lower-res i
                    ops = []
                    in_ch = channels[j]
                    for k in range(i - j):
                        out_ch = channels[i] if k == (i - j - 1) else in_ch
                        ops.append(
                            nn.Conv2d(
                                in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False
                            )
                        )
                        ops.append(nn.BatchNorm2d(out_ch))
                        if k != (i - j - 1):
                            ops.append(nn.ReLU(inplace=True))
                        in_ch = out_ch
                    fuse_row.append(nn.Sequential(*ops))
            self.fuse_layers.append(fuse_row)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_list):
        assert len(x_list) == self.num_branches, "Branch count mismatch"

        # per-branch processing
        y = [b(x) for b, x in zip(self.branches, x_list)]

        # fusion
        out = []
        for i in range(self.num_branches):
            fused = None
            tgt_h, tgt_w = y[i].shape[-2:]
            for j in range(self.num_branches):
                z = self.fuse_layers[i][j](y[j])
                if j > i:
                    z = F.interpolate(z, size=(tgt_h, tgt_w), mode="nearest")
                fused = z if fused is None else (fused + z)
            out.append(self.relu(fused))
        return out


class MiniHRNetBackbone(nn.Module):
    """
    Paper-aligned behavior (without external HRNet dependency):
    - input 299x299x3
    - two 3x3 stride-2 stem convolutions
    - parallel high-to-low resolution branches with repeated fusions
    - no final global pooling
    - output feature map fixed to 64x56x56
    """
    def __init__(self):
        super().__init__()

        # Stem: 299 -> 150 -> 75
        self.conv1 = ConvBNReLU(3, 64, k=3, s=2, p=1)
        self.conv2 = ConvBNReLU(64, 64, k=3, s=2, p=1)

        # Stage 1 (single high-res branch)
        self.stage1 = _make_branch(64, 64, num_blocks=2)

        # Transition to 2 branches
        self.transition1 = nn.ModuleList([
            nn.Identity(),  # branch 0: 64, 75x75
            nn.Sequential(  # branch 1: 128, 38x38
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ),
        ])
        self.stage2 = HRNetStage([64, 128])

        # Transition to 3 branches
        self.transition2 = nn.ModuleList([
            nn.Identity(),  # keep 64
            nn.Identity(),  # keep 128
            nn.Sequential(  # new branch: 256, 19x19
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
        ])
        self.stage3 = HRNetStage([64, 128, 256])

        # Transition to 4 branches
        self.transition3 = nn.ModuleList([
            nn.Identity(),  # keep 64
            nn.Identity(),  # keep 128
            nn.Identity(),  # keep 256
            nn.Sequential(  # new branch: 512, 10x10
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
        ])
        self.stage4 = HRNetStage([64, 128, 256, 512])

        # Fuse multi-scale concatenation -> 64 channels
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(64 + 128 + 256 + 512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.stage1(x)

        x_list = [self.transition1[0](x), self.transition1[1](x)]
        y_list = self.stage2(x_list)

        x_list = [self.transition2[0](y_list[0]), self.transition2[1](y_list[1]), self.transition2[2](y_list[1])]
        y_list = self.stage3(x_list)

        x_list = [
            self.transition3[0](y_list[0]),
            self.transition3[1](y_list[1]),
            self.transition3[2](y_list[2]),
            self.transition3[3](y_list[2]),
        ]
        y_list = self.stage4(x_list)

        # Upsample all branches to highest resolution branch
        h, w = y_list[0].shape[-2:]
        y0 = y_list[0]
        y1 = F.interpolate(y_list[1], size=(h, w), mode="nearest")
        y2 = F.interpolate(y_list[2], size=(h, w), mode="nearest")
        y3 = F.interpolate(y_list[3], size=(h, w), mode="nearest")
        y = torch.cat([y0, y1, y2, y3], dim=1)
        y = self.fuse_proj(y)  # [B,64,h,w]

        # Paper target feature map for downstream capsule fusion
        y = F.interpolate(y, size=(56, 56), mode="bilinear", align_corners=False)
        return y  # [B,64,56,56]


class LBPBlock(nn.Module):
    """
    Local binary pattern-style maps (8-neighbor comparator) per channel.
    """
    def __init__(self):
        super().__init__()
        self.offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        b, c, h, w = x.shape
        xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
        center = xp[:, :, 1:1 + h, 1:1 + w]
        outs = []
        for dy, dx in self.offsets:
            yy = 1 + dy
            xx = 1 + dx
            nbr = xp[:, :, yy:yy + h, xx:xx + w]
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


class HRNetLBPCapsNet(nn.Module):
    """
    Paper-aligned architecture (practical implementation):
      1) HRNet-style high-resolution backbone (no final pooling)
      2) LBP texture extraction in HSV + YCbCr spaces
      3) Feature fusion: HRNet(64x56x56) + texture(6x56x56) -> 70x56x56
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
    ):
        super().__init__()
        self.num_classes = num_classes

        # HRNet-like feature extractor (no external dependency)
        self.hrnet = MiniHRNetBackbone()
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

        lbp_hsv = self.lbp(hsv)      # [B,24,H,W]
        lbp_ycc = self.lbp(ycbcr)    # [B,24,H,W]

        lbp_hsv = F.interpolate(lbp_hsv, size=(56, 56), mode="nearest")
        lbp_ycc = F.interpolate(lbp_ycc, size=(56, 56), mode="nearest")

        lbp = 0.5 * (lbp_hsv + lbp_ycc)
        tex = self.texture_reduce(lbp)  # [B,6,56,56]
        return tex

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hr = self.hrnet(x)                 # [B,64,56,56]
        tex = self._texture_features(x)    # [B,6,56,56]

        feat = self.fusion(torch.cat([hr, tex], dim=1))  # [B,64,56,56]

        pri = self.primary_caps(feat)      # [B,N_in,8]
        dig = self.digit_caps(pri)         # [B,num_classes,16]

        lengths = torch.norm(dig, dim=-1)  # [B,num_classes]
        return lengths


# Backward-compatible alias used elsewhere in project
SegCapsCNN = HRNetLBPCapsNet


if __name__ == "__main__":
    model = HRNetLBPCapsNet(num_classes=2)
    dummy = torch.rand(2, 3, 299, 299)
    out = model(dummy)
    print("Output shape:", out.shape)  # expected: [2, 2]