# Grounded VQG Reproduction

Architecture-faithful reproduction of **"Automatic Generation of Grounded Visual
Questions"** (Zhang, Qu, You, Yang, Zhang -- IJCAI 2017, [arXiv:1612.06530](https://arxiv.org/abs/1612.06530)).

Goal: reproduce the paper's model exactly as specified, swapping out only the pieces
that are deprecated/unrunnable today (old Torch/Lua DenseCap), and documenting every
place the paper leaves a value unspecified.

## Status

- [x] Core architecture implemented (`src/`), matching Sec 3.1-3.2's equations.
- [x] Local smoke test passes (`tests/smoke_test.py`, tiny synthetic data, no downloads needed).
- [ ] Full training on VQA v1 / Visual7W -- run via `notebooks/colab_train.ipynb` on Colab Pro.

## What's faithful vs. substituted vs. gap-filled

Every source file's docstring cites the exact paper section/equation it implements.
Summary:

| Piece | Status |
|---|---|
| Question-type selector (LSTM+softmax, 6 types) | **As-specified** (Sec 3.1) |
| Caption encoder + correlation layer (Linear 600->300 + PReLU) | **As-specified** -- encoder hidden size is *pinned* to 300 because that's the only value consistent with the paper's stated 300x600 correlation linear layer, see `src/correlation.py` |
| LSTM decoder | **As-specified** (Sec 3.2), hidden size gap-filled |
| Kneser-Ney bigram joint decoding | **As-specified**, `d=0.75` (paper-given) |
| EM-style latent-caption training (Eq. 5-7) | **As-specified**, `alpha=0.75` (paper-given) |
| Adam, batch size 64, 128/64 epochs | **As-specified** (Sec 4.4) |
| VGG-16 frozen image features | As-specified role, different weight source (`torchvision` vs. the original Caffe port) |
| **DenseCap** (region proposals + captions) | **Substituted**: original is Torch7/Lua and unrunnable today (see "Fidelity notes" below) -> [`soloist97/densecap-pytorch`](https://github.com/soloist97/densecap-pytorch). This only affects the *input* caption pool's diversity, not the VQG model itself. |
| Learning rate, LSTM hidden sizes (except the pinned one above), vocab cutoff, dropout, gradient clip, bigram interpolation weight `beta`, multi-word type-prefix rule | **Gap-filled** -- the paper never states these. All defaults live in `configs/default.yaml` with inline comments; nothing is hardcoded in model code. |
| NeuralTalk2 baseline | Not reproduced (this repo trains only the paper's own model) |

## Fidelity notes: why DenseCap is substituted

The original DenseCap (`jcjohnson/densecap`, Torch7/Lua) is not officially dead, but:
- Its own OS target, `ubuntu:16.04`, no longer has working apt mirrors.
- Its install script itself needs manual patching (references a package renamed years ago).
- Its `stnbhwd` dependency (the spatial-transformer module for region resampling) is
  **actually gone** -- the maintainer's entire GitHub account was deleted. Only unverified
  community mirrors remain.

A sandboxed Docker build attempt (Ubuntu 18.04, CPU-only) confirmed the stack is "alive
but rotting" -- getting past even the first install step already needs a manual patch,
with the genuinely broken `stnbhwd` dependency still ahead, untested. Full original-stack
reproduction was judged a multi-hour-to-day archaeology project with no guarantee of
success, so this repo uses `densecap-pytorch` instead. See git history / conversation
notes for the full build log if you want to attempt the original stack yourself.

**Practical effect of this substitution:** `densecap-pytorch` initializes region
proposals from a Faster R-CNN (COCO/ImageNet-pretrained) backbone rather than DenseCap's
original jointly-trained fully-convolutional localization layer. Expect somewhat less
diverse region/caption coverage (more "canonical object" boxes, fewer relational/attribute
captions) -- this mainly affects the paper's *coverage/diversity* metrics (Fig. 3), not
the architecture being tested.

## Fidelity notes: why we train densecap-pytorch ourselves instead of using its pretrained checkpoint

`densecap-pytorch`'s own README points to a pretrained checkpoint on OneDrive/BaiduYun.
Both turned out to be unreachable in practice:
- **OneDrive:** the share link resolves to "item deleted or don't have permission" even
  when signed into a Microsoft account (tried both the `1drv.ms` short link and the
  resolved `onedrive.live.com/?cid=...` URL that another user confirmed working in
  [the repo's issue #2](https://github.com/soloist97/densecap-pytorch/issues/2) back in 2022).
- **BaiduYun:** the file (1.24GB) exceeds what Baidu Netdisk allows anonymous
  browser downloads without a China-verified account.
- No Hugging Face mirror, GitHub Release, or other re-host exists (checked directly).

So `notebooks/colab_train.ipynb` (cell 6) trains `densecap-pytorch` itself on Visual
Genome, using the repo's own `preprocess.py` and a lightly patched copy of its
`train.py` (`scripts/densecap_train_patched.py`). This does **not** add a new fidelity
compromise beyond the substitution above -- it's the same code, same architecture,
same target dataset the author used; the only difference from downloading their weights
is normal training-run seed variance. The patch itself makes two changes, both purely
about robustness for a multi-hour Colab job, not the training procedure:
1. Replaces NVIDIA Apex (`from apex import amp`) with native `torch.cuda.amp` -- Apex
   predates PyTorch's built-in mixed-precision support and needs a from-source CUDA
   build, an unnecessary extra failure point.
2. Adds checkpoint-resume support, which upstream has none of (it only saves on a
   validation-mAP improvement or at the very end of all epochs -- a Colab disconnect
   mid-run could otherwise lose everything). Saves an unconditional checkpoint to
   Drive at the end of every epoch and auto-resumes from it on restart.

Visual Genome's own download (images + region annotations) is a normal, public,
no-login academic dataset host (`cs.stanford.edu`, `homes.cs.washington.edu`) --
all four URLs in `scripts/download_visual_genome.sh` were verified live via `curl -I`
before being used, not guessed.

## Repo layout

```
src/                  core architecture modules (one file per paper component)
scripts/              data prep: download VQA/Visual7W/COCO/VisualGenome/GloVe,
                       train+run DenseCap, build manifest
configs/default.yaml  all hyperparameters, tagged paper-given vs. gap-filled
eval/evaluate.py      BLEU/METEOR/ROUGE-L via pycocoevalcap, precision + recall/coverage curves
tests/smoke_test.py   end-to-end pipeline check on tiny synthetic data (no downloads)
notebooks/colab_train.ipynb   full pipeline for Colab Pro: data prep -> train DenseCap
                               -> generate captions -> train VQG model -> eval
```

## Quickstart (local, smoke test only)

```bash
pip install torch torchvision pyyaml numpy pillow
python -m tests.smoke_test
```

## Full training (Colab Pro)

Open `notebooks/colab_train.ipynb` in Colab, set a GPU runtime, fill in the config cell
(GitHub repo URL, dataset choice, DenseCap checkpoint path), and run top to bottom. See
the notebook's own markdown cells for what each stage does and how long it takes.

## Evaluation

`eval/evaluate.py` reproduces the paper's Fig. 3 methodology exactly: for each
image, generate N in {1..6} questions, then compute both precision (best-matching
reference per generated question) and recall/coverage (best-matching generated
question per reference) using BLEU-1..4/METEOR/ROUGE-L. Do not expect exact number
matches to the paper's published curves -- see the substitution/gap-fill table above
for why.
