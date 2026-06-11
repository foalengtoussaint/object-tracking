"""One-shot DINOv2 object matching — prototype + cosine-threshold detection.

A frozen DINOv2 backbone turns an image into a grid of patch features. An
object is described by a single *prototype* vector: the mean of the L2-normed
patch features that fall inside its mask. Detection on a new frame is then a
cosine-similarity map against that prototype, an adaptive percentile threshold,
and a connected-component pick.

No training is involved — onboarding an object is one mask + one forward pass.
Because DINOv2 features are web-scale and frozen, the prototype generalizes
across lighting/background far better than a small fine-tuned detector, which is
the reason this is worth having alongside the YOLO student.

This is a clean-room reimplementation of the method (DINOv2 features → prototype
→ adaptive threshold → connected components, with a PerSAM-style argmax
fallback). The DINOv2 backbone itself is loaded from torch.hub (Apache-2.0).

Output polygons use this repo's YOLO-seg convention: normalized (x, y) pairs,
class 0, so they drop straight into pseudo_label.py / the seed datasets.

Examples
--------
    import cv2, swissdino_lib as sd
    model = sd.load_dino("vit_b")

    # Onboard from a frame + its SAM mask (HxW bool/uint8, image resolution).
    fmap = sd.extract_feature_map(cv2.imread("seed.jpg"), model)
    proto, thresh = sd.build_prototype(fmap, sd.mask_to_patch(mask, fmap.shape[0]))

    # Detect on a query frame.
    fmap_q = sd.extract_feature_map(cv2.imread("query.jpg"), model)
    patch_mask, score = sd.detect(fmap_q, proto, thresh)
    poly = sd.mask_patch_to_polygon(patch_mask, img_w, img_h)
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch

# DINOv2 hub names by short alias. vit_b is the default — best speed/robustness
# balance for offline labeling.
HUB_NAMES = {"vit_s": "dinov2_vits14", "vit_b": "dinov2_vitb14", "vit_l": "dinov2_vitl14"}

# Square side the image is resized to before the backbone. Must be a multiple of
# the patch size (14); 448 -> 32x32 patch grid.
IMAGE_RESIZE = 448

# ImageNet normalization (DINOv2 was trained with these).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Percentile for the adaptive threshold (see build_prototype).
DEFAULT_PERCENTILE = 5


def _device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=3)
def load_dino(variant: str = "vit_b", device: str | None = None):
    """Load a frozen DINOv2 backbone (cached per variant/device).

    variant: 'vit_s' | 'vit_b' | 'vit_l'. Returns an eval-mode torch module.
    """
    hub_name = HUB_NAMES.get(variant)
    if hub_name is None:
        raise ValueError(f"unknown variant {variant!r}; choose {list(HUB_NAMES)}")
    dev = _device(device)
    print(f"loading DINOv2 {hub_name} on {dev} ...")
    model = torch.hub.load("facebookresearch/dinov2", hub_name)
    model.to(dev).eval()
    print("DINOv2 ready.")
    return model


@torch.no_grad()
def extract_feature_map(img_bgr: np.ndarray, model, resize: int = IMAGE_RESIZE) -> np.ndarray:
    """BGR image (cv2) -> [P, P, D] grid of patch features, P = resize // 14.

    The returned features are NOT normalized; build_prototype/detect normalize
    as needed.
    """
    dev = next(model.parameters()).device
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (resize, resize), interpolation=cv2.INTER_CUBIC)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev)  # [1,3,H,W]

    out = model.forward_features(tensor)
    tokens = out["x_norm_patchtokens"][0]  # [P*P, D]
    p = resize // model.patch_size
    fmap = tokens.reshape(p, p, -1).float().cpu().numpy()
    return fmap


def mask_to_patch(mask: np.ndarray, patch_num: int,
                  binarize_threshold: float = 0.5) -> np.ndarray:
    """Resize a pixel-space mask (HxW, bool/uint8/float) to a [P, P] bool grid."""
    m = np.asarray(mask)
    if m.dtype == np.bool_:
        m = m.astype(np.uint8) * 255
    elif m.max() <= 1.0:
        m = (m * 255).astype(np.uint8)
    else:
        m = m.astype(np.uint8)
    resized = cv2.resize(m, (patch_num, patch_num), interpolation=cv2.INTER_AREA)
    return resized > (binarize_threshold * 255)


def _cosine_map(feature_map: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    """[P,P,D] features, [D] unit prototype -> [P,P] cosine similarity in [-1,1]."""
    normed = feature_map / (np.linalg.norm(feature_map, axis=-1, keepdims=True) + 1e-8)
    return normed @ prototype


def build_prototype(feature_map: np.ndarray, patch_mask: np.ndarray,
                    percentile: int = DEFAULT_PERCENTILE) -> tuple[np.ndarray, float]:
    """Compute the object prototype and detection threshold from one masked frame.

    prototype = unit-normalized mean of the masked patch features.
    threshold = the larger of (a) the high-percentile cosine over *background*
    patches and (b) the low-percentile cosine over *object* patches — high
    enough to reject most background while still admitting the object.

    Returns (prototype [D], threshold float).
    """
    if patch_mask.sum() < 1:
        raise ValueError("empty patch mask — object did not cover any patch")

    feat_dim = feature_map.shape[2]
    proto = feature_map[patch_mask].reshape(-1, feat_dim).mean(axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-8)

    self_cos = _cosine_map(feature_map, proto)
    bg = self_cos[~patch_mask]
    fg = self_cos[patch_mask]
    bg_high = np.percentile(bg, 100 - percentile) if bg.size else -1.0
    fg_low = np.percentile(fg, percentile)
    threshold = float(max(bg_high, fg_low))
    return proto.astype(np.float32), threshold


def detect(feature_map: np.ndarray, prototype: np.ndarray,
           threshold: float) -> tuple[np.ndarray, float]:
    """Find the best-matching object region in a query feature map.

    Thresholds the cosine map, splits it into connected components, and returns
    the component whose pooled cosine best matches the prototype. If nothing
    clears the threshold, falls back to the single best-matching patch (PerSAM).

    Returns (patch_mask [P,P] bool, score float in [-1,1]).
    """
    cos = _cosine_map(feature_map, prototype)
    hits = cos >= threshold
    if not hits.any():
        # PerSAM fallback: the single most prototype-like patch.
        best = cos == cos.max()
        return best, float(cos.max())

    num, labels = cv2.connectedComponents(hits.astype(np.uint8), connectivity=8)
    best_mask, best_score = None, -np.inf
    feat_dim = feature_map.shape[2]
    for comp in range(1, num):
        comp_mask = labels == comp
        pooled = feature_map[comp_mask].reshape(-1, feat_dim).mean(axis=0)
        pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
        score = float(pooled @ prototype)
        if score > best_score:
            best_mask, best_score = comp_mask, score
    return best_mask, best_score


def score_crop(feature_map: np.ndarray, bbox_xyxy, prototype: np.ndarray) -> float:
    """Appearance score (cosine) of a pixel-space bbox against the prototype.

    For gating an externally-proposed detection (e.g. a YOLO box) by whether it
    actually looks like the onboarded object. bbox is in the same pixel
    coordinates as the image the feature map came from is irrelevant — patches
    are located by fraction, so pass the bbox in the *original image* pixel
    space along with that image's (w, h) via the closure below.

    bbox_xyxy: (x_min, y_min, x_max, y_max) as fractions of the image in [0,1].
    Returns the mean-pooled cosine of the patches inside the box.
    """
    p = feature_map.shape[0]
    x0, y0, x1, y1 = bbox_xyxy
    px0, py0 = int(np.floor(x0 * p)), int(np.floor(y0 * p))
    px1, py1 = int(np.ceil(x1 * p)), int(np.ceil(y1 * p))
    px0, py0 = max(0, px0), max(0, py0)
    px1, py1 = min(p, max(px0 + 1, px1)), min(p, max(py0 + 1, py1))
    region = feature_map[py0:py1, px0:px1].reshape(-1, feature_map.shape[2])
    pooled = region.mean(axis=0)
    pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
    return float(pooled @ prototype)


def mask_patch_to_polygon(patch_mask: np.ndarray, img_w: int, img_h: int,
                          epsilon_frac: float = 0.0015) -> list[list[float]] | None:
    """Patch-grid bool mask -> normalized polygon [[x,y],...] in image coords.

    Upsamples the patch mask to image resolution, takes the largest external
    contour, simplifies it, and normalizes to [0,1]. Matches the polygon format
    written by sam_label_server.py / pseudo_label.py. Returns None if empty.
    """
    if patch_mask is None or not patch_mask.any():
        return None
    full = cv2.resize(patch_mask.astype(np.uint8) * 255, (img_w, img_h),
                      interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea).astype(np.float32)
    if len(best) < 3:
        return None
    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, epsilon_frac * peri, True).reshape(-1, 2)
    return [[float(x) / img_w, float(y) / img_h] for x, y in approx]
