# BFLDL Implementation Guide (Paper-Aligned Update)

This guide documents the updated implementation that replaces the earlier basic convolutional feature extractor with an **HRNet-style high-resolution backbone**, and keeps the **LBP + Capsule routing** components consistent with the paper’s design intent.

Paper reference:  
**“A Novel Blockchain-Based Deepfake Detection Method Using Federated and Deep Learning Models” (2024)**

---

## 1) What was changed

### Previous state
- The original model used a relatively basic CNN front-end plus capsule layers.
- Feature extraction did not fully reflect the HRNet-centered description in the paper.

### Updated state
- `model.py` now uses a **paper-traceable architecture**:
  1. **HRNet-style backbone** (high-resolution, multi-branch, repeated fusion, no final global pooling)
  2. **Texture branch using LBP** over **HSV** and **YCbCr** projections
  3. **Feature fusion** of HRNet and texture maps
  4. **Capsule classifier** with **routing-by-agreement**
- `SegCapsCNN` remains as a backward-compatible alias to avoid breaking existing imports:
  - `SegCapsCNN = HRNetLBPCapsNet`

---

## 2) Paper-to-code traceability

Below is how key paper statements are reflected in the codebase.

## A. Input normalization and size standardization
Paper notes:
- Normalize by min-max equation: `q_norm = (q - q_min) / (q_max - q_min)`
- Standardized image size around `299 x 299 x 3`
- Lanczos resize used for spatial normalization
- Random augmentation (rotation, horizontal/vertical flip, color changes)

Code alignment:
- `preprocess.py`:
  - `normalize_data(...)` implements Eq. (1)-style min-max scaling.
  - `DeepfakePreprocessor` uses:
    - Lanczos resize (`InterpolationMode.LANCZOS`)
    - Random crop to `299x299`
    - Random rotation
    - Horizontal + vertical flipping
    - Color jitter
  - Deterministic eval transform is also provided.

## B. HRNet replacement for basic CNN
Paper notes:
- HRNet is used to maintain high-resolution representations.
- High/low-resolution branches run in parallel with frequent feature fusion.
- Final pooling from original HRNet is removed to preserve spatial detail for capsule usage.
- Described feature map target around `64 x 56 x 56` before capsule integration.

Code alignment:
- `model.py`:
  - `MiniHRNetBackbone` implements:
    - Two stride-2 stem convolutions.
    - Parallel multi-resolution branches.
    - Repeated fusion across branches (upsample/downsample + summation).
    - No final classification head / global pooling.
  - Output is explicitly resized to `64 x 56 x 56` for downstream capsule stack.

## C. Texture analysis with LBP + color spaces
Paper notes:
- Texture information is extracted with LBP from color spaces (HSV and YCbCr).
- Texture cues are fused with HRNet-derived structural features.

Code alignment:
- `model.py`:
  - `LBPBlock` computes local binary pattern-style maps using 8-neighbor comparisons.
  - Input RGB tensor is projected to:
    - HSV (`_rgb_to_hsv`)
    - YCbCr (`_rgb_to_ycbcr`)
  - LBP maps from both spaces are combined and reduced to compact texture channels.
  - Texture channels are fused with HRNet features prior to capsule projection.

## D. Capsule network and routing-by-agreement
Paper notes:
- Classification relies on capsule layers and dynamic routing.
- Softmax coupling coefficients and squash nonlinearity are core operations.

Code alignment:
- `model.py`:
  - `CapsuleLayer`:
    - Uses learned transform matrix `W`
    - Applies routing logits `b_ij`
    - Uses softmax coupling over output capsules
    - Performs iterative agreement updates
  - `squash(...)` is implemented for capsule activation.
  - Output class probabilities are represented by capsule vector lengths.

## E. Segmentation / slice handling
Paper notes:
- Mentions using 2D slices from 3D volumes, especially XZ or YX planes.

Code alignment:
- `preprocess.py`:
  - `segment_volume(...)` supports:
    - `XZ` slicing
    - `YX`/`XY` extraction
  - Works with tensor/ndarray inputs and standardizes to canonical video tensor layout.

---

## 3) Files impacted

- `model.py`
  - Replaced basic convolutional feature extractor with HRNet-style backbone.
  - Added texture + capsule integration in a single model pipeline.
  - Preserved `SegCapsCNN` alias for compatibility.

- `preprocess.py`
  - Expanded to include paper-consistent preprocessing pipeline:
    - spatial normalization (Lanczos resize + size handling)
    - signal normalization (min-max)
    - augmentation
    - 2D slice extraction helpers

No changes required in orchestrator logic for federated flow (`main.py`, `blockchain_fl.py`) because model interfaces were preserved.

---

## 4) Practical notes and assumptions

- This is a **paper-aligned engineering implementation**, not a verbatim reproduction of every hidden training detail.
- The HRNet portion is implemented as a **minimal HRNet-style architecture** to preserve:
  - high-resolution stream maintenance,
  - parallel multi-scale branches,
  - repeated fusion behavior,
  - spatial detail retention for capsules.
- LBP histogram specifics are approximated with differentiable tensor operations and channel reduction suitable for training integration.
- Capsule routing is implemented in the canonical dynamic-routing form and outputs class capsule lengths.

---

## 5) Runtime/training expectations

- Model memory usage is significantly higher than the original basic CNN due to:
  - larger multi-branch backbone,
  - high-resolution feature preservation,
  - dense capsule routing nodes (`56*56*primary_capsules`).
- If training becomes unstable or too heavy:
  - reduce `primary_capsules`,
  - reduce batch size,
  - use mixed precision,
  - tune routing iterations (`routing_iters`).

---

## 6) Suggested validation checklist

To verify correctness in your environment:

1. **Shape checks**
   - Input: `[B, 3, 299, 299]`
   - HRNet feature: `[B, 64, 56, 56]`
   - Output logits/probabilities tensor: `[B, 2]`

2. **Preprocessing checks**
   - Confirm min-max normalization maps values to `[0,1]`.
   - Confirm train augmentation and eval deterministic paths differ as intended.

3. **Capsule checks**
   - Verify routing iterations execute without dimension mismatch.
   - Verify output is non-negative capsule lengths.

4. **End-to-end FL smoke run**
   - `run_bfldl_workflow()` should complete one federated round with aggregation and sync.

---

## 7) Summary

The implementation now reflects the paper’s intended pipeline much more closely:

**HRNet-style structural features + LBP texture cues + capsule routing classifier**, integrated into the federated blockchain workflow without breaking existing call sites.