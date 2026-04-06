# BFLDL Implementation Guide (Step-by-Step + Code Dissection)

This guide explains **exactly what happens in the code**, in order, so you can trace the pipeline from input frames all the way to federated aggregation.

Paper reference:  
**“A Novel Blockchain-Based Deepfake Detection Method Using Federated and Deep Learning Models” (2024)**

---

## 1) Repository flow at a glance

Your runtime path is:

1. `main.py` creates the global model (`SegCapsCNN`, alias to `HRNetLBPCapsNet`)
2. `main.py` creates clients and blockchain orchestrator
3. Each client trains locally using `ClientNode.train_locally(...)`
4. Clients send model weights to blockchain orchestrator
5. Orchestrator performs FedAvg and updates global model
6. Updated global model is broadcast back to all clients

Model internals are in `model.py`; preprocessing utilities are in `preprocess.py`.

---

## 2) Model architecture flow (forward pass in plain English)

The model class used by training is:

- `SegCapsCNN` → alias of `HRNetLBPCapsNet`

Given input tensor `x` with shape `[B, 3, 299, 299]`, `HRNetLBPCapsNet.forward(x)` does:

1. **HRNet branch**
   - Calls `self.hrnet(x)` (`TimmHRNetBackbone`)
   - Produces `hr` shaped `[B, 64, 56, 56]`

2. **Texture branch**
   - Calls `self._texture_features(x)`
   - Converts RGB to HSV + YCbCr
   - Applies LBP-like extractor on each space
   - Resizes both LBP tensors to `56x56`, averages them
   - Reduces channels to `6`
   - Produces `tex` shaped `[B, 6, 56, 56]`

3. **Feature fusion**
   - Concatenates `[hr, tex]` on channel axis → `[B, 70, 56, 56]`
   - Applies `self.fusion` (3x3 conv + BN + ReLU) → `[B, 64, 56, 56]`

4. **Primary capsules**
   - `self.primary_caps(feat)` converts conv feature map into capsules
   - Output shape becomes `[B, N_in, 8]`, where:
     - `N_in = 56 * 56 * primary_capsules`
     - default `primary_capsules = 8`
     - so `N_in = 25088`

5. **Digit/class capsules with routing**
   - `self.digit_caps(pri)` (`CapsuleLayer`)
   - Dynamic routing for `routing_iters` rounds (default 3)
   - Output shape `[B, num_classes, 16]` (default classes: 2)

6. **Class score extraction**
   - L2 norm of each class capsule vector:
   - `lengths = torch.norm(dig, dim=-1)`
   - final output shape: `[B, 2]` (Real/Fake scores)

---

## 3) `model.py` function-by-function dissection

## A) `squash(inputs, axis=-1, eps=1e-9)`
Purpose:
- Capsule nonlinearity.
- Shrinks short vectors toward 0 and long vectors toward length ~1.

Math in code:
- `norm_sq = sum(inputs^2)`
- `scale = norm_sq / (1 + norm_sq)`
- output = `scale * inputs / sqrt(norm_sq + eps)`

Why:
- Capsule probability is represented by vector length.

---

## B) `class CapsuleLayer(nn.Module)`

Implements dynamic routing between:
- input capsules: `N_in`, dim `D_in`
- output capsules: `N_out`, dim `D_out`

### `__init__(...)`
Key fields:
- `num_capsules` = `N_out` (e.g., 2 classes)
- `num_route_nodes` = `N_in`
- `in_channels` = `D_in`
- `out_channels` = `D_out`
- `num_iterations` = routing rounds

Learned weight:
- `self.W` shape `[N_out, N_in, D_out, D_in]`

### `forward(x)`
Expected input:
- `x` shape `[B, N_in, D_in]`

Steps:
1. Validate input dimensions.
2. Compute predicted outputs:
   - `u_hat = einsum("oidc,bic->boid", W, x)`
   - shape `[B, N_out, N_in, D_out]`
3. Initialize routing logits:
   - `b_ij = zeros([B, N_out, N_in])`
4. Loop routing iterations:
   - `c_ij = softmax(b_ij, dim=1)` (coupling coeffs)
   - `s_j = sum_i(c_ij * u_hat)` over input capsules
   - `v_j = squash(s_j)`
   - if not last iteration:
     - agreement = dot(`u_hat`, `v_j`)
     - update logits: `b_ij += agreement`
5. Return `v_j` (`[B, N_out, D_out]`)

---

## C) `class LBPBlock(nn.Module)`

Purpose:
- Local Binary Pattern style texture coding.

### `forward(x)`
Input:
- `x` shape `[B, C, H, W]`

Steps:
1. Reflect-pad by 1 pixel.
2. For each of 8 neighbor offsets:
   - compare neighbor vs center (`nbr >= center`)
   - cast to numeric map.
3. Concatenate 8 maps per channel:
   - output shape `[B, 8*C, H, W]`

---

## D) `class ConvPrimaryCaps(nn.Module)`

Purpose:
- Convert fused conv features into primary capsules.

### `__init__(in_channels, num_capsules=8, capsule_dim=8)`
- 1x1 projection conv to `num_capsules * capsule_dim` channels.

### `forward(x)`
Input:
- `[B, C, H, W]`

Steps:
1. 1x1 conv → `[B, num_capsules*capsule_dim, H, W]`
2. Reshape and permute to isolate capsule dimension.
3. Flatten spatial + capsule index into `N_in`.
4. Apply `squash`.

Output:
- `[B, H*W*num_capsules, capsule_dim]`

---

## E) `class TimmHRNetBackbone(nn.Module)`

Purpose:
- External HRNet feature extractor using `timm`.

### `__init__(model_name="hrnet_w18", pretrained=False)`
- Creates model via:
  - `timm.create_model(..., features_only=True, out_indices=(0,1,2,3,4))`
- Gets channel counts from `feature_info`.
- Creates `fuse_proj` to compress concatenated multi-scale features to 64 channels.

### `forward(x)`
Input:
- `[B,3,H,W]` (typically 299x299)

Steps:
1. Compute padding to next multiple-of-32 for H and W.
2. Replicate-pad input if needed (`x_safe`).
   - This avoids HRNet branch size mismatches on non-aligned sizes.
3. Run HRNet features:
   - returns list of feature maps at multiple scales.
4. Upsample all feature maps to highest spatial resolution among them.
5. Concatenate on channels.
6. Project via `fuse_proj` to 64 channels.
7. Resize to `[56,56]`.

Output:
- `[B,64,56,56]`

---

## F) `class HRNetLBPCapsNet(nn.Module)`

Main end model class.

### `__init__(...)`
Constructs:
- `self.hrnet` (`TimmHRNetBackbone`)
- `self.lbp` (`LBPBlock`)
- `self.texture_reduce` (24 → 16 → 6 channels)
- `self.fusion` (64+6 → 64)
- `self.primary_caps` (`ConvPrimaryCaps`)
- `self.digit_caps` (`CapsuleLayer`)

Key calculated value:
- `route_nodes = 56 * 56 * primary_capsules`

### `_rgb_to_hsv(x)`
- Approximate RGB→HSV conversion in tensor form.
- Input `[B,3,H,W]`, output `[B,3,H,W]`.

### `_rgb_to_ycbcr(x)`
- RGB→YCbCr conversion.
- Input `[B,3,H,W]`, output `[B,3,H,W]`.

### `_texture_features(x)`
Steps:
1. Clamp input to `[0,1]`.
2. Convert to HSV and YCbCr.
3. LBP both:
   - each `[B,24,H,W]` because `8 * 3 channels`.
4. Resize both to `[56,56]`.
5. Average two LBP tensors.
6. Reduce channels to 6 via `self.texture_reduce`.

Output:
- `[B,6,56,56]`

### `forward(x)`
Already detailed in Section 2; returns `[B,2]`.

---

## G) `SegCapsCNN = HRNetLBPCapsNet`
Why it exists:
- Backward compatibility with older imports in other files.

---

## 4) `preprocess.py` dissection (data pipeline utilities)

## A) `normalize_data(q, eps=1e-12)`
Implements paper Eq. (1):

- `q_norm = (q - q_min) / (q_max - q_min)`

Supports both:
- `torch.Tensor`
- `numpy.ndarray`

---

## B) `@dataclass AugmentationConfig`
Contains tunables for training augmentation:
- resize size (default 320x320)
- random crop size (default 299x299)
- horizontal/vertical flip probabilities
- max rotation
- color jitter params

---

## C) `class DeepfakePreprocessor`

### `__init__(target_size=(299,299), cfg=None)`
Creates two transform pipelines:

1. **Train transform**
   - Resize with Lanczos
   - Random crop to 299x299
   - H/V flips
   - Random rotation
   - Color jitter
   - ToTensor

2. **Eval transform**
   - Deterministic resize to target size
   - ToTensor

---

### `preprocess_image(img, train=True)`
- Applies train or eval transform.
- Applies min-max normalization with `normalize_data`.
- Returns `[3,H,W]` tensor.

---

### `preprocess_frame_tensor(frame, train=True)`
Accepts frame tensor in either:
- `[C,H,W]` or `[H,W,C]`
- uint8 or float

Steps:
1. Validate dimensionality
2. Rearrange to CHW if needed
3. Cast to float
4. If range looks like `[0,255]`, scale down to `[0,1]`
5. Convert to PIL
6. Delegate to `preprocess_image(...)`

---

### `preprocess_video_frames(frames, train=True)`
Input:
- sequence of PIL images or tensors

Steps:
1. preprocess each frame
2. stack

Output:
- `[T,3,299,299]`

---

### `_to_video_tensor(video_frames)`
Normalizes video container formats into canonical `[T,C,H,W]`.

Supports:
- torch tensor
- numpy array
- list/tuple of per-frame tensors/arrays

Also handles channel-last format conversion.

---

### `segment_volume(video_frames, plane="XZ", center_index=None, return_all_channels=True)`
Implements 2D slicing notion from paper.

Steps:
1. Canonicalize to `[T,C,H,W]`
2. Convert to float and normalize
3. If `plane == "XZ"`:
   - fix y-index (center by default)
   - extract `vol[:, :, y_idx, :]` and permute to `[C,T,W]`
4. If `plane in {"YX","XY"}`:
   - return full per-frame spatial planes `[T,C,H,W]`

Optional:
- return grayscale-like average if `return_all_channels=False`.

---

## 5) `blockchain_fl.py` dissection (federated + blockchain simulation)

## A) `class BlockchainOrchestrator`

### `__init__(initial_model)`
Stores:
- `global_model`
- `local_updates` list
- `ledger` list (simple records)

### `register_update(client_id, model_weights)`
Creates transaction-like update dict and appends to `local_updates`.

### `aggregate_consensus()`
Implements FedAvg:

1. If no updates, return
2. For each parameter key:
   - average across all submitted client state dicts
3. Load averaged weights into `global_model`
4. Clear local updates
5. Append ledger message

### `broadcast_model()`
Returns deep-copied global model for clients to sync.

---

## B) `class ClientNode`

### `__init__(client_id, local_model)`
- stores local model
- creates Adam optimizer (`lr=0.001`)

### `train_locally(dummy_data, dummy_labels, epochs=1)`
For each epoch:
1. zero grad
2. forward pass
3. loss = `mse_loss(outputs, labels)`
4. backward
5. optimizer step

### `get_weights()`
- returns `state_dict()`

### `sync_global_model(global_model)`
- loads global model state dict into local model

---

## 6) `main.py` dissection (orchestration)

## `run_bfldl_workflow()`

Step-by-step exactly as implemented:

1. Create global model:
   - `global_model = SegCapsCNN()`
2. Create orchestrator:
   - `blockchain = BlockchainOrchestrator(global_model)`
3. Create preprocessor:
   - `preprocessor = DeepfakePreprocessor(target_size=(299,299))`
4. Create clients (deep copies of global):
   - two clients: `Resource_A`, `Resource_B`
5. Create dummy batch:
   - input `[1,3,299,299]`
   - label `[[0.0, 1.0]]`
6. For each client:
   - local train for 1 epoch
   - register weights to blockchain
7. Aggregate on blockchain (FedAvg)
8. Broadcast global model
9. Each client syncs to global
10. print completion status

---

## 7) Tensor-shape checkpoints you can print for debugging

Recommended checkpoints inside `HRNetLBPCapsNet.forward`:

- input `x`: `[B,3,299,299]`
- `hr = self.hrnet(x)`: `[B,64,56,56]`
- `tex = self._texture_features(x)`: `[B,6,56,56]`
- fused `feat`: `[B,64,56,56]`
- `pri`: `[B,56*56*primary_capsules,8]`
- `dig`: `[B,2,16]`
- output `lengths`: `[B,2]`

---

## 8) End-to-end training signal summary

At each client:

1. Input image batch → model output `[B,2]`
2. Compare with labels (`[B,2]`) via MSE
3. Backprop updates local params
4. Send local state dict to orchestrator
5. Orchestrator averages params (FedAvg)
6. New global params distributed to all clients

So even though data is local, model parameters are shared/aggregated.

---

## 9) External dependencies used by this implementation

- `torch`
- `torchvision`
- `numpy`
- `timm`
- `Pillow`

---

## 10) Practical notes for you while reading the code

- If you want to inspect routing behavior, start in `CapsuleLayer.forward`.
- If you want to inspect texture cues, start in `_texture_features` + `LBPBlock`.
- If you want to inspect HRNet behavior, start in `TimmHRNetBackbone.forward`.
- If you want to inspect FL logic, start in:
  - `ClientNode.train_locally`
  - `BlockchainOrchestrator.aggregate_consensus`
  - `main.run_bfldl_workflow`

---

## 11) One-line mental model

**Input frame → preprocess normalization/augment → HRNet structural features + LBP texture features → capsule routing classifier → local client training → FedAvg on blockchain orchestrator → synced global model.**