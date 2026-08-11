"""Pretrained, COCO-trained object detector used as a reliability cross-check --
NOT part of the paper's architecture, a separate, opt-in inference-time addition (see
src/generate.py's `object_suppression_bias`).

Why: DenseCap was trained from scratch on a Colab budget (mAP ~0.09, see
notebooks/colab_train.ipynb's discussion), and separately, the VQG decoder itself has
been observed generating confident, image-independent guesses that are literally COCO
class names -- "bird", "mouse", "cake", "vase", "scissors", "banana" all showed up as
wrong answers across several real test images, and all eighty COCO classes are exactly
the kind of frequent object words a model trained on COCO-derived VQA data would learn
a strong generic prior toward. torchvision's fasterrcnn_resnet50_fpn ships pretrained
on COCO to ~37 mAP -- no training, no API key required, and it's architecturally a
classic two-stage CNN detector (backbone + region proposal network + ROI heads), not a
transformer, in the same spirit as everything else in this "lightweight" reproduction.
"""
from typing import List, Optional, Set, Tuple

import torch
import torchvision
from PIL import Image
from torchvision import transforms

# Basic named-color reference points (RGB) for _dominant_color -- nearest-neighbor
# lookup, not a learned model. Deliberately coarse (11 common color words) since the
# goal is a word a human would plausibly use ("red", "white"), not colorimetric
# precision.
NAMED_COLORS = {
    "red": (196, 30, 40), "orange": (230, 126, 34), "yellow": (241, 196, 15),
    "green": (39, 130, 60), "blue": (41, 91, 180), "purple": (120, 50, 140),
    "pink": (230, 130, 180), "brown": (110, 70, 40), "black": (30, 30, 30),
    "white": (235, 235, 230), "gray": (130, 130, 130),
}

# Index-aligned with torchvision's pretrained fasterrcnn_resnet50_fpn output labels
# (standard COCO detection category list, including the placeholder background/N-A
# entries at the indices COCO's own category ids skip).
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Informal/alternate words for a COCO class that don't match via the plain
# last-word heuristic in build_object_suppression_mask -- e.g. "airplane"'s own last
# (only) word is "airplane", but DenseCap/natural language overwhelmingly say "plane"
# instead, which slipped through the suppression check entirely as an unrecognized
# word (confirmed in a real test run: "who took this plane ?" on an image with no
# plane at all). Deliberately small and hand-picked, not a general synonym system --
# stays lightweight, only covers cases actually observed or obviously likely.
COCO_CLASS_SYNONYMS = {
    "airplane": ["airplane", "plane"],
    "motorcycle": ["motorcycle", "motorbike"],
    "couch": ["couch", "sofa"],
    "tv": ["tv", "television"],
    "cell phone": ["phone", "cellphone"],
}

_model = None  # loaded once per process, reused across calls -- same reasoning as
                # other frozen feature extractors in this codebase (src/image_features.py)


def _get_model(device: str):
    global _model
    if _model is None:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        _model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
    return _model


@torch.no_grad()
def _detect(image_path: str, confidence_threshold: float,
            device: str = None) -> Tuple[List[Tuple[str, float, Tuple[float, float, float, float]]], Image.Image]:
    """Runs the detector once, returns ([(class_name, score, box), ...], image) for
    every detection above confidence_threshold (excluding background/N-A), one entry
    per detected box -- not deduped, since e.g. two separate people are each a
    legitimate detection. box is (x1, y1, x2, y2) in pixel coordinates. Also returns
    the loaded image itself so callers doing pixel-level work (_dominant_color) don't
    need to reopen the file. detect_objects() and detect_objects_as_candidates() both
    derive from this single call so neither duplicates the (cheap, but non-zero)
    forward pass."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _get_model(device)
    img = Image.open(image_path).convert("RGB")
    tensor = transforms.ToTensor()(img).to(device)
    pred = model([tensor])[0]

    detections = []
    for label, score, box in zip(pred["labels"].tolist(), pred["scores"].tolist(), pred["boxes"].tolist()):
        if score >= confidence_threshold:
            name = COCO_INSTANCE_CATEGORY_NAMES[label]
            if name not in ("__background__", "N/A"):
                detections.append((name, score, tuple(box)))
    return detections, img


def detect_objects(image_path: str, confidence_threshold: float = 0.5, device: str = None) -> Set[str]:
    """Returns the set of COCO class names detected in the image above
    confidence_threshold, e.g. {"person", "banana", "cell phone"} -- used by
    src/generate.py's object_suppression_bias."""
    detections, _ = _detect(image_path, confidence_threshold, device)
    return {name for name, _, _ in detections}


def _nearest_color_name(rgb: Tuple[float, float, float]) -> str:
    best_name, best_dist = None, float("inf")
    for name, ref in NAMED_COLORS.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, ref))
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name


def _dominant_color(img: Image.Image, box: Tuple[float, float, float, float]) -> Optional[str]:
    """Average color inside the box, mapped to the nearest name in NAMED_COLORS --
    plain pixel arithmetic, not a model. Downsamples to 16x16 first since we only need
    a rough average, not per-pixel precision."""
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.width), min(y2, img.height)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img.crop((x1, y1, x2, y2)).resize((16, 16))
    pixels = list(crop.getdata())
    avg = tuple(sum(p[i] for p in pixels) / len(pixels) for i in range(3))
    return _nearest_color_name(avg)


def _position_phrase(box: Tuple[float, float, float, float], img_w: int, img_h: int) -> Optional[str]:
    """Coarse position in the frame ("left", "top left", ...) from the box center --
    plain geometry. Returns None for a roughly-centered box, since "in the center"
    for every other object isn't informative enough to be worth stating."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h = "left" if cx < img_w / 3 else ("right" if cx > 2 * img_w / 3 else None)
    v = "top" if cy < img_h / 3 else ("bottom" if cy > 2 * img_h / 3 else None)
    if v and h:
        return f"{v} {h}"
    return v or h


def _size_phrase(box: Tuple[float, float, float, float], img_w: int, img_h: int) -> Optional[str]:
    """"large"/"small" if the box takes up a notably large/small fraction of the
    image, None for anything in between -- plain geometry, not a model."""
    x1, y1, x2, y2 = box
    area_ratio = ((x2 - x1) * (y2 - y1)) / max(img_w * img_h, 1)
    if area_ratio > 0.25:
        return "large"
    if area_ratio < 0.03:
        return "small"
    return None


def detect_objects_as_candidates(image_path: str, confidence_threshold: float = 0.5,
                                  device: str = None) -> List[dict]:
    """Same detections as detect_objects, enriched with color/size/position and
    formatted as candidate dicts ({"caption": "a large red truck on the left",
    "confidence": score}) so they can be merged directly into DenseCap's own
    candidate pool (see notebooks/colab_train.ipynb cell 10).

    DenseCap produces richer natural-language captions ("a woman holding a cell
    phone") but, trained from scratch, is unreliable about WHAT's actually in the
    image. This detector is the reverse trade-off: reliable about object identity
    (COCO-trained to ~37 mAP) but a bare noun phrase ("a truck") has much less to
    work with than DenseCap's captions. The color/position/size attributes here are
    NOT a model -- they're deterministic pixel/geometry math computed directly from
    the detector's own verified box, so they add descriptive richness with zero
    hallucination risk (nothing here can be "wrong" the way a generative caption can,
    short of the underlying detection itself being wrong).

    Pooling both sources lets confidence-weighted sampling draw on whichever is
    actually right for a given region, and -- unlike object_suppression_bias, which
    corrects individual word choices *after* the decoder has already committed to a
    direction -- gives the decoder itself better raw material to generate a fluent
    sentence around, rather than fighting its output word by word."""
    detections, img = _detect(image_path, confidence_threshold, device)
    candidates = []
    for name, score, box in detections:
        parts = []
        size = _size_phrase(box, img.width, img.height)
        if size:
            parts.append(size)
        color = _dominant_color(img, box)
        if color:
            parts.append(color)
        parts.append(name)
        caption = "a " + " ".join(parts)
        position = _position_phrase(box, img.width, img.height)
        if position:
            caption += f" in the {position}"
        candidates.append({"caption": caption, "confidence": score})
    return candidates


def _box_overlap_stats(box_a: Tuple[float, float, float, float],
                        box_b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Returns (fraction of box_a inside box_b, fraction of box_b inside box_a) --
    plain intersection-area geometry."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    contain_a_in_b = inter / area_a if area_a > 0 else 0.0
    contain_b_in_a = inter / area_b if area_b > 0 else 0.0
    return contain_a_in_b, contain_b_in_a


# "on" is only asserted for class pairs where a physically plausible support/mount
# relationship actually exists -- pure 2D box overlap can't distinguish "genuinely
# stacked" (a person on a horse) from "just overlaps in the projection because the
# camera happens to line them up" (a tv behind a person's head reads as "contained"
# in 2D despite being nowhere near the person physically). Confirmed in a real test:
# "a tv on a person", "a book on a person" for a desk/office scene where nothing was
# actually stacked. Deliberately small and hand-picked, not exhaustive.
PLAUSIBLE_ON_PAIRS = {
    frozenset({"person", "horse"}), frozenset({"person", "bicycle"}),
    frozenset({"person", "motorcycle"}), frozenset({"person", "bench"}),
    frozenset({"person", "chair"}), frozenset({"person", "couch"}),
    frozenset({"person", "skateboard"}), frozenset({"person", "surfboard"}),
    frozenset({"person", "snowboard"}), frozenset({"person", "skis"}),
    frozenset({"person", "bed"}), frozenset({"cat", "couch"}), frozenset({"cat", "bed"}),
    frozenset({"cat", "chair"}), frozenset({"dog", "couch"}), frozenset({"dog", "bed"}),
    frozenset({"dog", "chair"}), frozenset({"bird", "bench"}),
}


def _edge_gap(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Straight-line distance between the nearest edges of two boxes -- 0 if they
    overlap or touch. More robust than center-to-center distance when box sizes
    differ a lot (e.g. a large horse and a small nearby truck can have distant
    centers despite their edges nearly touching)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return (dx ** 2 + dy ** 2) ** 0.5


def _relation_phrase(name_a: str, box_a: Tuple[float, float, float, float], area_a: float,
                      name_b: str, box_b: Tuple[float, float, float, float], area_b: float,
                      img_w: int, img_h: int) -> Optional[str]:
    """"a {smaller} on a {larger}" if one box is mostly contained within the other
    (e.g. a person mostly overlapping a horse/bench -- a decent proxy for "riding"/
    "sitting on" without needing to actually recognize the action), "a {a} near a
    {b}" if their edges are close but they don't overlap much, else None. Pure box
    geometry, no model -- same reliability guarantee as the color/position/size
    enrichment: this can only be as wrong as the underlying detections themselves."""
    if name_a == name_b:
        return None
    contain_a_in_b, contain_b_in_a = _box_overlap_stats(box_a, box_b)
    plausible_on = frozenset({name_a, name_b}) in PLAUSIBLE_ON_PAIRS

    if plausible_on and contain_a_in_b > 0.5 and area_a < area_b:
        return f"a {name_a} on a {name_b}"
    if plausible_on and contain_b_in_a > 0.5 and area_b < area_a:
        return f"a {name_b} on a {name_a}"

    if contain_a_in_b < 0.1 and contain_b_in_a < 0.1:
        diag = (img_w ** 2 + img_h ** 2) ** 0.5
        gap = _edge_gap(box_a, box_b) / max(diag, 1)
        if gap < 0.1:
            return f"a {name_a} near a {name_b}"

    return None


def detect_relations_as_candidates(image_path: str, confidence_threshold: float = 0.5,
                                    device: str = None, max_detections: int = 8,
                                    max_relations: int = 6) -> List[dict]:
    """Simple relational captions ("a person on a horse", "a horse near a truck")
    built from pairwise box geometry between the same detections detect_objects_as_
    candidates() enriches individually -- see notebooks/colab_train.ipynb cell 10.

    Right now each detection is described in isolation; this is what lets the
    candidate pool express how two things relate, which is a big part of what makes a
    question feel "contextually aware" rather than generic ("what color is the X")
    -- without a new model, without training, just geometry on boxes the detector
    already produced and already verified are there.

    Limited to the top `max_detections` by confidence (bounds the O(n^2) pair count)
    and returns at most `max_relations` (highest-confidence pairs first), so a busy
    image with many detections doesn't flood the candidate pool with low-value
    combinations."""
    detections, img = _detect(image_path, confidence_threshold, device)
    top = sorted(detections, key=lambda d: -d[1])[:max_detections]

    candidates = []
    seen_pairs = set()
    for i in range(len(top)):
        name_a, score_a, box_a = top[i]
        area_a = max(box_a[2] - box_a[0], 0) * max(box_a[3] - box_a[1], 0)
        for j in range(i + 1, len(top)):
            name_b, score_b, box_b = top[j]
            area_b = max(box_b[2] - box_b[0], 0) * max(box_b[3] - box_b[1], 0)
            phrase = _relation_phrase(name_a, box_a, area_a, name_b, box_b, area_b, img.width, img.height)
            if phrase is None:
                continue
            key = tuple(sorted((name_a, name_b)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            candidates.append({"caption": phrase, "confidence": min(score_a, score_b)})

    candidates.sort(key=lambda c: -c["confidence"])
    return candidates[:max_relations]


def build_object_suppression_mask(detected_objects: Set[str], vocab, device: str) -> torch.Tensor:
    """1.0 at vocab positions for the head noun of every COCO class NOT present in
    detected_objects, 0.0 elsewhere -- used by src/generate.py to actively suppress
    the decoder's generic "safe" object guesses when a real detector doesn't confirm
    them for this specific image. Only the last word of each class name is used as
    its "head noun" (e.g. "cell phone" -> "phone") -- a simple, deliberately
    approximate heuristic; a few classes share a head noun ("dog" / "hot dog") and get
    treated the same, an acceptable imprecision for a safety-net bias, not a precision
    requirement."""
    mask = torch.zeros(len(vocab.idx2word), device=device)
    for name in COCO_INSTANCE_CATEGORY_NAMES:
        if name in ("__background__", "N/A") or name in detected_objects:
            continue
        words = COCO_CLASS_SYNONYMS.get(name, [name.split()[-1]])
        for w in words:
            idx = vocab.word2idx.get(w)
            if idx is not None:
                mask[idx] = 1.0
    return mask
