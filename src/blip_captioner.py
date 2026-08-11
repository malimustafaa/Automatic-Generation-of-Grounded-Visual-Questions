"""BLIP (Salesforce, via the `transformers` library) used as a drop-in replacement
for DenseCap's role -- generating natural-language captions -- while VGG-16 (paper
Sec 3.2's own specified image-feature source, src/image_features.py) stays completely
untouched for the image *feature vector* role. NOT part of the paper's architecture:
like DenseCap itself (Sec 3.1, "the DenseCap model of Johnson et al., 2016"), the
caption source is an external, swappable upstream input to the paper's own VQG model
(src/model.py, src/correlation.py, src/decoder.py) -- none of those files change here.

Why: DenseCap trained from scratch on a Colab budget measured ~0.09 mAP -- genuinely
poor, and repeatedly introduced wrong/irrelevant content across real test images.
BLIP ships pretrained (COCO + web-scale image-text pairs) and produces dramatically
more reliable, natural captions, at a real weight cost (~247M params for
blip-image-captioning-base -- see notebooks/colab_train.ipynb's weight/fidelity
discussion) that VGG-16/Faster R-CNN don't have.

BLIP captions one image (or crop) at a time, unlike DenseCap's own multi-region
output -- caption_regions_as_candidates() reuses Faster R-CNN's region proposals
(src/object_detector.py's detections, already computed there for the
object-suppression cross-check) as the crop source, running BLIP on each region to
reconstruct DenseCap's "many candidate captions per image" format (Sec 3.1's
confidence-weighted candidate set C_i).
"""
from typing import List

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from .object_detector import _detect

_MODEL_NAME = "Salesforce/blip-image-captioning-base"
_processor = None
_model = None  # loaded once per process, reused across calls -- same pattern as
                # object_detector.py's _get_model


def _get_blip(device: str):
    global _processor, _model
    if _model is None:
        _processor = BlipProcessor.from_pretrained(_MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(_MODEL_NAME).to(device).eval()
    return _processor, _model


@torch.no_grad()
def _generate_caption(img: Image.Image, device: str) -> str:
    processor, model = _get_blip(device)
    inputs = processor(img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)


def caption_whole_image(image_path: str, device: str = None) -> dict:
    """One BLIP caption for the entire image. confidence is fixed at 1.0 -- BLIP
    doesn't expose a comparable per-caption confidence score the way DenseCap/
    Faster R-CNN's detections do; treated as a high-trust default since it's
    describing the whole frame, not asserting a specific narrow claim."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(image_path).convert("RGB")
    return {"caption": _generate_caption(img, device), "confidence": 1.0}


def caption_regions_as_candidates(image_path: str, confidence_threshold: float = 0.3,
                                   max_regions: int = 8, device: str = None) -> List[dict]:
    """Runs BLIP on each of Faster R-CNN's top `max_regions` detected boxes (by
    confidence), replacing DenseCap's per-region captioning entirely. Reuses
    object_detector._detect() rather than re-running Faster R-CNN separately, since
    the object-suppression cross-check already needs that same detection pass --
    avoids loading/running the detector twice.

    confidence_threshold defaults lower than detect_objects's own 0.5 -- BLIP
    generates its description directly from the crop's pixels rather than trusting
    the COCO class label, so a region worth describing doesn't need as high a
    detection confidence as one being asserted as a specific, suppressible class
    name would."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    detections, img = _detect(image_path, confidence_threshold, device)
    top = sorted(detections, key=lambda d: -d[1])[:max_regions]

    candidates = []
    for _name, score, box in top:
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img.width), min(y2, img.height)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img.crop((x1, y1, x2, y2))
        candidates.append({"caption": _generate_caption(crop, device), "confidence": score})
    return candidates


@torch.no_grad()
def caption_image_batched(image_path: str, max_regions: int = 4,
                           region_confidence_threshold: float = 0.3, device: str = None) -> List[dict]:
    """Same content as caption_whole_image() + caption_regions_as_candidates()
    combined (one whole-image caption plus BLIP on the top max_regions detected
    boxes), but issues ONE batched BLIP call for all of them together instead of
    max_regions+1 separate generate() calls. Meaningfully faster per image (much
    better GPU utilization per call) -- used by scripts/generate_candidates_blip.py,
    which needs to do this for every image in a whole dataset, not just one at a time
    like the notebook demo cares about."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = _get_blip(device)

    detections, img = _detect(image_path, region_confidence_threshold, device)
    top = sorted(detections, key=lambda d: -d[1])[:max_regions]

    crops, scores = [img], [1.0]
    for _name, score, box in top:
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img.width), min(y2, img.height)
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(img.crop((x1, y1, x2, y2)))
        scores.append(score)

    inputs = processor(images=crops, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    captions = processor.batch_decode(out, skip_special_tokens=True)

    return [{"caption": cap.strip(), "confidence": conf} for cap, conf in zip(captions, scores)]
