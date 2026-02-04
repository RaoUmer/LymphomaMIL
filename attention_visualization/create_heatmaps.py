#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create attention heatmaps from a WSI using TRIDENT + a Gated MIL Aggregator.

Fixes included:
- Safe-first checkpoint loading (PyTorch 2.6+ weights_only behavior)
- State-dict key cleaning & strict=False load
- Pass device **string** to TRIDENT ("cuda:0"/"cpu")
- Shim for visualize_heatmap expecting wsi.read_region_pil(...)
- Clamp vis_level to available levels
- Cap TRIDENT worker count (if supported by your TRIDENT version)
- Correct features shape [N, D] and attention extraction
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

from trident import OpenSlideWSI, visualize_heatmap
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory as patch_encoder_factory


# ---------------------------
#   Compatibility shims
# ---------------------------

def ensure_read_region_pil(wsi):
    """
    TRIDENT visualize_heatmap() expects wsi.read_region_pil(loc, level, size) -> PIL.Image.
    Some OpenSlideWSI versions only expose read_region. This shim adds read_region_pil
    by aliasing or wrapping the underlying backend.
    """
    if hasattr(wsi, "read_region_pil"):
        return wsi

    if hasattr(wsi, "read_region"):
        # OpenSlideWSI.read_region usually returns a PIL image already
        wsi.read_region_pil = wsi.read_region
        return wsi

    backend = getattr(wsi, "slide", None)
    if backend is None or not hasattr(backend, "read_region"):
        raise AttributeError(
            "WSI lacks read_region/read_region_pil and no .slide backend found."
        )

    def _read_region_pil(loc, level, size):
        return backend.read_region(loc, level, size).convert("RGB")

    wsi.read_region_pil = _read_region_pil
    return wsi


# ---------------------------
#   MIL model (Gated Attention)
# ---------------------------

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_tasks=1):
        super().__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        # x: [N, L]
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)            # [N, D]
        A = self.attention_c(A) # [N, n_tasks], here n_tasks=1
        return A, x


class GMA(nn.Module):
    def __init__(self, ndim=1024, gate=True, size_arg="big", dropout=False, n_classes=2, n_tasks=1):
        super().__init__()
        self.size_dict = {"small": [ndim, 512, 256], "big": [ndim, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_tasks=1)
        fc.append(attention_net)

        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)

        initialize_weights(self)

    def get_sign(self, h):
        A, h = self.attention_net(h)  # h: [N, H]
        w = self.classifier.weight.detach()
        sign = torch.mm(h, w.t())
        return sign

    def forward(self, h, attention_only=False):
        """
        h: [N, feat_dim]
        If attention_only=True: returns raw attention scores per patch (1D length N).
        """
        A, h = self.attention_net(h)  # A: [N, 1]
        A = torch.transpose(A, 1, 0)  # A: [1, N]
        if attention_only:
            return A[0]               # [N] (pre-softmax)

        # Full forward (not used for heatmap, but kept here)
        A = F.softmax(A, dim=1)       # normalize across patches
        M = torch.mm(A, h)            # [1, H]
        logits = self.classifier(M)   # [1, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        return {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'A': A}


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()


# ---------------------------
#   Robust checkpoint loader (PyTorch 2.6+)
# ---------------------------

def _clean_state_dict_keys(sd):
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in sd.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        if nk.startswith('model.'):
            nk = nk[len('model.'):]
        if nk.startswith('net.'):
            nk = nk[len('net.'):]
        new_sd[nk] = v
    return new_sd


def load_model(ckpt_path, model, map_location='cpu'):
    """
    Safe-first loading. If that fails, falls back to weights_only=False
    (ONLY if you trust the checkpoint source).
    """
    import numpy as np
    import torch

    # Allow-list numpy scalar for safe unpickling (PyTorch 2.6+)
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass

    # Try safe load
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except Exception:
        # Trusted fallback (unsafe for untrusted files)
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # Extract state_dict
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            sd = ckpt['state_dict']
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            sd = ckpt['model']
        else:
            sd = ckpt
    else:
        sd = ckpt

    sd = _clean_state_dict_keys(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load_model] strict=False. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if len(missing) < 30:
            print("  Missing:", missing)
        if len(unexpected) < 30:
            print("  Unexpected:", unexpected)

    model.eval()
    return model


# ---------------------------
#   Inference helper
# ---------------------------

@torch.inference_mode()
def run_inference(model, patch_features, apply_softmax=True):
    """
    patch_features: Tensor [N, D] on device
    Returns per-patch attention scores: Tensor [N]
    """
    att_raw = model(patch_features, attention_only=True)  # [N]
    if apply_softmax:
        return torch.softmax(att_raw, dim=0)
    return att_raw


# ---------------------------
#   Main pipeline
# ---------------------------

def main():
    # ----- Paths & params -----/
    checkpoint = '/path/to/trained_model/checkpoint_latest_kfold0.pth'
    slide_path = '/path/to/slides/sample_slide.svs' 
    job_dir = '/path/to/save/heatmap_viz/'
    os.makedirs(job_dir, exist_ok=True)

    target_mag = 40
    patch_size = 224
    feature_dim = 1536  # UNI v2 features
    desired_vis_level = 8  # will be clamped to valid levels later
    num_top_patches_to_save = 20 # Number of high-score topk patches to save

    # ----- Device handling -----
    if torch.cuda.is_available():
        cuda_idx = torch.cuda.current_device()
        device_str = f"cuda:{cuda_idx}"   # TRIDENT expects a string
    else:
        device_str = "cpu"
    device_torch = torch.device(device_str) 

    print(f"[info] Using device: {device_str}")

    # ----- Load MIL aggregator -----
    model = GMA(ndim=feature_dim, n_classes=5)
    model = load_model(checkpoint, model, map_location=device_torch).to(device_torch)

    # ----- Open slide and ensure compatibility -----
    slide = OpenSlideWSI(slide_path=slide_path, lazy_init=False)
    slide = ensure_read_region_pil(slide)

    # Clamp vis level to valid range
    level_dims = getattr(slide, "level_dimensions", None)
    if isinstance(level_dims, (list, tuple)) and len(level_dims) > 0:
        vis_level = min(desired_vis_level, len(level_dims) - 1)
    else:
        vis_level = 0
    print('vis_level:', vis_level)

    # ----- Tissue segmentation -----
    segmentation_model = segmentation_model_factory("hest")
    # Optional caps for workers/batch if your TRIDENT exposes these args
    seg_kwargs = {
        "num_workers": min(8, os.cpu_count() or 8),
        "batch_size": 8
    }
    allowed_seg_kwargs = {}
    try:
        allowed = set(slide.segment_tissue.__code__.co_varnames)
        allowed_seg_kwargs = {k: v for k, v in seg_kwargs.items() if k in allowed}
    except Exception:
        pass

    geojson_contours = slide.segment_tissue(
        segmentation_model=segmentation_model,
        job_dir=job_dir,
        device=device_str,  # pass string, not torch.device
        **allowed_seg_kwargs
    )
    if geojson_contours is None or (isinstance(geojson_contours, list) and len(geojson_contours) == 0):
        raise RuntimeError("No tissue regions found; adjust segmentation parameters.")

    # ----- Patch coordinates -----
    coords_path = slide.extract_tissue_coords(
        target_mag=target_mag,
        patch_size=patch_size,
        save_coords=job_dir,
        overlap=0,
    )
    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"coords file not found at {coords_path}")

    # ----- Patch feature extraction (UNI v2) -----
    patch_encoder = patch_encoder_factory("uni_v2").eval().to(device_torch)  # foundation model name
    feat_kwargs = {
        "num_workers": min(8, os.cpu_count() or 8),
        "batch_size": 128,
        "pin_memory": torch.cuda.is_available()
    }
    allowed_feat_kwargs = {}
    try:
        allowed = set(slide.extract_patch_features.__code__.co_varnames)
        allowed_feat_kwargs = {k: v for k, v in feat_kwargs.items() if k in allowed}
    except Exception:
        pass

    patch_features_path = slide.extract_patch_features(
        patch_encoder=patch_encoder,
        coords_path=coords_path,
        save_features=os.path.join(job_dir, "features_uni_v2"),
        device=device_str,  # pass string
        **allowed_feat_kwargs
    )
    if not os.path.exists(patch_features_path):
        raise FileNotFoundError(f"features file not found at {patch_features_path}")

    # ----- Load coords & features -----
    with h5py.File(patch_features_path, 'r') as f:
        coords = f['coords'][:]
        feats = f['features'][:]
        coords_attrs = dict(f['coords'].attrs)

    if coords is None or len(coords) == 0:
        raise RuntimeError("No patch coords loaded.")
    if feats is None or len(feats) == 0:
        raise RuntimeError("No patch features loaded.")

    feats = torch.from_numpy(feats).float().to(device_torch)   # [N, D]
    assert feats.ndim == 2, f"Expected 2D [N, D], got {feats.shape}"
    assert feats.shape[1] == feature_dim, f"Expected D={feature_dim}, got {feats.shape[1]}"

    # ----- Inference: per-patch attention -----
    attention = run_inference(model, feats, apply_softmax=True)  # [N]

    # Determine patch_size at level 0 (fallback if missing)
    patch_size_level0 = int(coords_attrs.get('patch_size_level0', patch_size))

    # ----- Heatmap visualization -----
    heatmap_save_path = visualize_heatmap(
        wsi=slide,
        scores=attention.detach().cpu().numpy(),  # 1D array length N
        coords=coords,
        vis_level=vis_level,
        patch_size_level0=patch_size_level0,
        normalize=True,
        num_top_patches_to_save=num_top_patches_to_save,
        output_dir=job_dir
    )
    print(f"[done] Heatmap saved to: {heatmap_save_path}")


if __name__ == '__main__':
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    main()
