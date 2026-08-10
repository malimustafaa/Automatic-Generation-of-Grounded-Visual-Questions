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
from typing import Set

import torch
import torchvision
from PIL import Image
from torchvision import transforms

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

_model = None  # loaded once per process, reused across calls -- same reasoning as
                # other frozen feature extractors in this codebase (src/image_features.py)


def _get_model(device: str):
    global _model
    if _model is None:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        _model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
    return _model


@torch.no_grad()
def detect_objects(image_path: str, confidence_threshold: float = 0.5, device: str = None) -> Set[str]:
    """Returns the set of COCO class names detected in the image above
    confidence_threshold, e.g. {"person", "banana", "cell phone"}. Excludes the
    background/N-A placeholder entries."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _get_model(device)
    img = Image.open(image_path).convert("RGB")
    tensor = transforms.ToTensor()(img).to(device)
    pred = model([tensor])[0]

    detected = set()
    for label, score in zip(pred["labels"].tolist(), pred["scores"].tolist()):
        if score >= confidence_threshold:
            name = COCO_INSTANCE_CATEGORY_NAMES[label]
            if name not in ("__background__", "N/A"):
                detected.add(name)
    return detected


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
        head_noun = name.split()[-1]
        idx = vocab.word2idx.get(head_noun)
        if idx is not None:
            mask[idx] = 1.0
    return mask
