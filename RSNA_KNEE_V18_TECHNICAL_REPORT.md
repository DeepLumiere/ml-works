# Technical Report: Multi-Representation Ensemble Architecture & Optimization Strategies for RSNA Knee Abnormality Detection (`rsna-knee-v18.ipynb`)

---

## Executive Summary

This report presents a comprehensive technical breakdown of the multi-representation ensemble pipeline implemented in `rsna-knee-v18.ipynb` for the **RSNA Knee Abnormality Detection** competition. The pipeline predicts 12 target abnormality findings:
`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, and `Fracture`.

---

## 1. Tri-Arm Multi-Representation Architecture Overview

The system combines three distinct deep learning architectural families—**CoAtNet (CNN + Attention Hybrid)**, **DINOv3 (Self-Supervised ViT-S/16)**, and **RadImageNet (Medically Pretrained ResNet-50)**—into a metadata-conditioned probability ensemble:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                          RSNA KNEE V18 TRI-ARM SYSTEM DIAGRAM                               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

                                ┌─────────────────────────────┐
                                │      DICOM Knee Volume      │
                                └──────────────┬──────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
   ┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
   │    Arm 1: CoAtNet     │       │    Arm 2: DINOv3      │       │   Arm 3: RadImageNet  │
   │       (Raptor)        │       │      (ViT-S/16)       │       │      (ResNet-50)      │
   │ 64 Slices, 140mm Crop │       │ 6 Slots, 2.5D RGB     │       │  3 Fluid Slots (2.5D) │
   └───────────┬───────────┘       └───────────┬───────────┘       └───────────┬───────────┘
               │                               │                               │
               └───────────────────────────────┼───────────────────────────────┘
                                               ▼
                         ┌───────────────────────────────────────────┐
                         │   Metadata-Conditioned Blending (MoE)     │
                         │    (Slot Coverage + Probability Space)    │
                         └─────────────────────┬─────────────────────┘
                                               ▼
                         ┌───────────────────────────────────────────┐
                         │    Dynamic Hierarchical Calibration       │
                         │    (Fluid Boost + Anatomical Coupling)    │
                         └─────────────────────┬─────────────────────┘
                                               ▼
                         ┌───────────────────────────────────────────┐
                         │          Final Submission CSV             │
                         └───────────────────────────────────────────┘
```

---

## 2. Pipeline Component Details & Mathematical Formulations

### Arm 1: CoAtNet (Raptor) 3D Spatial Sliding Window Volume Classifier

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                             ARM 1: COATNET (RAPTOR) PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

  [ 64-Slice DICOM Stack ] ──► [ Spatial Dilation [c-2, c, c+2] ] ──► [ Sliding 3-Slice Windows ]
                                                                             │
  [ Per-Finding Logits ] ◄── [ Einsum Query Attention Pool ] ◄── [ CoAtNet Encoder ]
```

* **Spatial Dilation Window Extraction**:
  For DICOM volumes with depth $D \ge 5$, spatial dilation extracts slices $[c - 2, c, c + 2]$:
  $$\mathbf{W}_j = \text{Stack}\left(\mathbf{V}_{c_j - 2}, \mathbf{V}_{c_j}, \mathbf{V}_{c_j + 2}\right) \in \mathbb{R}^{3 \times 224 \times 224}$$
* **Einsum Query Attention Pooling**:
  $$A_{b, k, n} = \text{Softmax}\left(\mathbf{W}_{\text{att}} \tanh(\mathbf{W}_1 \mathbf{H}_{b, k} + \mathbf{b}_1)\right)$$
  $$\mathbf{F}_{\text{pooled}, b, n} = \sum_{k} A_{b, k, n} \mathbf{H}_{b, k}$$
  $$\mathbf{\hat{Y}}_{b, n} = \sigma\left(\sum_{f} \mathbf{F}_{\text{pooled}, b, n, f} \cdot \mathbf{W}_{\text{cls}, n, f} + \mathbf{b}_{\text{cls}, n}\right)$$

---

### Arm 2: DINOv3 (ViT-S/16) 6-Slot Token Embeddings

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              ARM 2: DINOV3 (VIT-S/16) PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

  [ 6 Slot DICOM Cache ] ──► [ Sequential 2.5D RGB Channels ] ──► [ ViT-S/16 Backbone ]
                                                                          │
  [ Target Probabilities ] ◄── [ Slot Token Query Attention ] ◄── [ CLS Token Concatenation ]
```

* **Sequential 2.5D Input**:
  Target 3-slice group sequential RGB stacking:
  $$\mathbf{X}_m = \mathbf{X}_{[:, [0, 1, 2], H, W]} \in \mathbb{R}^{(B \cdot S) \times 3 \times 518 \times 518}$$
* **Slot Query Attention**:
  $$\mathbf{H} = \text{LayerNorm}(\mathbf{F}_{\text{concat}, b, s}) \mathbf{W}_{\text{proj}} \in \mathbb{R}^{B \times 6 \times 256}$$
  $$\mathbf{S} = \frac{\mathbf{Q} (\mathbf{H} + \mathbf{E}_{\text{slot}})^T}{\sqrt{256}} + \mathbf{M}_{\text{mask}}$$
  $$\mathbf{A} = \text{Softmax}(\mathbf{S}, \text{dim}=-1)$$
  $$\mathbf{\hat{Y}}_{b, n} = \sigma\left(\sum_{s} A_{b, n, s} H_{b, s} \cdot \mathbf{W}_{\text{out}, n} + b_{\text{out}, n}\right)$$

---

### Arm 3: RadImageNet (ResNet-50) Medical Feature Extractor

* **Encoder**: ResNet-50 backbone pretrained on millions of medical radiology scans.
* **Attention Head**: Multi-head query attention module with $\text{hidden}=512$ projection dimension targeting E10 fluid slots (`SAG_FLUID_FS`, `COR_FLUID_FS`, `AX_FLUID_FS`).

---

## 3. Preprocessing & Contrast Enhancements

### A. Extended Slice Band Sampling (`SLICE_BAND = (0.1, 0.9)`)
Retains 80% of DICOM stack depth, preserving peripheral slices where marginal osteophytes (OA) and meniscal root tears occur.

### B. Intensity Center-of-Mass (CoM) Dynamic Cropping
Calculates tissue Center-of-Mass using spatial OpenCV moments on Gaussian-blurred pixel arrays:
$$M_{ij} = \sum_{x, y} x^i y^j I(x, y)$$
$$c_x = \frac{M_{10}}{M_{00}}, \quad c_y = \frac{M_{01}}{M_{00}}$$
$$y_0 = \max\left(0, \min\left(c_y - \frac{\text{want}}{2}, h - \text{want}\right)\right), \quad x_0 = \max\left(0, \min\left(c_x - \frac{\text{want}}{2}, w - \text{want}\right)\right)$$

### C. Contrast Limited Adaptive Histogram Equalization (CLAHE) & Gamma Correction
Applies percentile clipping, gamma compression ($\mathbf{V}^{0.85}$), and CLAHE equalization (`clipLimit=2.0`, `tileGridSize=(8, 8)`) to accent trabecular bone edema and subtle fluid collections across $1.5\text{T}$ and $3.0\text{T}$ MRI scanners.

---

## 4. Key Bug Fixes & Resolved Bottlenecks

| Module | Issue / Bottleneck | Solution Implemented |
| :--- | :--- | :--- |
| **`_RadAttnHead`** | Size mismatch `[12, 512]` in checkpoint vs `[12, 256]` in model. | Updated `hidden=512` in `_RadAttnHead` and `run_radimagenet_arm`. |
| **`_RadAttnHead` State Dict** | $100\%$ missing keys on load. | Cleaned key prefixes via `sd_clean = {k.replace("module.", "").replace("model.", ""): v}`. |
| **`_DINOv3Model` Head** | Shape mismatch `torch.Size([1536])` vs `torch.Size([768])`. | Passed `dim=dim` ($384$) in `_DINOv3Model` so `_DINOv3Head` initializes `LayerNorm(768)`. |
| **DINOv3 2.5D Channels** | Spatial depth context collapse. | Updated `_DINOv3Model.forward` to pass sequential slices `x[:, [0, 1, 2]]`. |
| **Raptor Call** | Fatal `NameError` crash (`LAT_MAP`). | Passed defined `lat_map=lat_map` dictionary in `run_raptor_arm`. |

---

## 5. Ensembling & Hierarchical Calibration

### Direct Probability-Space Blending
Blends model outputs directly in bounded probability space $[0, 1]$:
$$\hat{P}_{i, c} = \sum_{a=1}^{A} W_{a, i, c} \cdot P_{a, i, c}$$

### Top-$K$ Window Pooling for Focal Findings
Requires spatial persistence across top-3 sliding windows for focal targets (`ACL`, `MCL`, `Meniscus`, `Fracture`, `Contusion`, `Baker's`):
$$\hat{P}_{\text{focal}, i, c} = \frac{1}{3} \sum_{k=1}^{3} P_{(K - k + 1), i, c}$$

### Dynamic Distribution-Adaptive Thresholds
Computes batch-wise distribution statistics (`mean + k * std`) dynamically:
$$\tau_{\text{fluid}} = \text{clip}\left(\mu_{\text{fluid}} + 0.5 \sigma_{\text{fluid}}, 0.10, 0.35\right)$$
$$\tau_{\text{contusion}} = \text{clip}\left(\mu_{\text{contusion}} + 1.2 \sigma_{\text{contusion}}, 0.50, 0.85\right)$$

---

## Conclusion

The `rsna-knee-v18.ipynb` pipeline integrates all three complementary model arms into a robust, leak-free submission pipeline that executes well within Kaggle's 10-hour GPU limit.
