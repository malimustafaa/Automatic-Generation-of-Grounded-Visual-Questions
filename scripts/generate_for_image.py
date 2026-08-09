"""Interactive helper: run DenseCap + the trained VQG model over a single arbitrary
image, for trying the pipeline out on your own pictures rather than the dataset.

Not part of the training/eval pipeline -- these are the two testable, reusable pieces
(DenseCap-on-one-image, VGG-16-feature-for-one-image); the Colab notebook's demo cell
wires them together with an interactive file-upload widget, which only makes sense
inside a live Colab session, not as a standalone script.
"""
import json
import os
import subprocess

import torch
from PIL import Image
from torchvision import transforms

from src.image_features import VGG16FeatureExtractor

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def run_densecap_single(densecap_repo: str, config_json: str, checkpoint: str,
                         image_path: str, result_dir: str, box_per_img: int = 20):
    """Same describe.py invocation as scripts/run_densecap.py (explicit --lut_path,
    cwd=densecap_repo, --verbose), just for exactly one image instead of a whole
    directory. Returns the candidate list in this project's format:
    [{"caption": str, "confidence": float}, ...].

    Checks image_path exists upfront with a specific error message, and captures
    subprocess output explicitly rather than letting it inherit stdout/stderr --
    confirmed in practice that a Jupyter/Colab notebook cell doesn't reliably display
    a subprocess's inherited output on failure (a describe.py crash showed only a bare
    CalledProcessError, none of the actual Python traceback from inside describe.py
    itself, even though the exact same command run from a Terminal showed it fine)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found at {image_path!r}. If this came from Colab's "
            "files.upload(), remember it saves into the notebook's current working "
            "directory, not necessarily /content/ -- use os.path.abspath(filename) "
            "rather than assuming a fixed prefix."
        )

    os.makedirs(result_dir, exist_ok=True)
    lut_path = os.path.join(densecap_repo, "data", "VG-regions-dicts-lite.pkl")
    cmd = [
        "python", os.path.join(densecap_repo, "describe.py"),
        "--config_json", config_json,
        "--model_checkpoint", checkpoint,
        "--img_path", image_path,
        "--result_dir", result_dir,
        "--box_per_img", str(box_per_img),
        "--lut_path", lut_path,
        "--verbose",
    ]
    result = subprocess.run(cmd, cwd=densecap_repo, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"describe.py failed (exit code {result.returncode}).\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

    with open(os.path.join(result_dir, "result.json")) as f:
        raw = json.load(f)
    # Single-image mode: result.json has exactly one key (the image path describe.py
    # was given), whatever its exact string form -- just take the one entry present.
    dets = next(iter(raw.values()))
    return [{"caption": d["cap"], "confidence": d["score"]} for d in dets]


def extract_single_image_feature(image_path: str, device: str = None) -> torch.Tensor:
    """Same frozen VGG-16 extractor as scripts/extract_image_features.py, for one image."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16FeatureExtractor(out_dim=300).to(device).eval()
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        feat = model(PREPROCESS(img).unsqueeze(0).to(device)).squeeze(0).cpu()
    return feat
