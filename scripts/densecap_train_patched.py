"""Patched copy of soloist97/densecap-pytorch's train.py.

Two changes from upstream, both purely about *robustness of the training run*, not
the model architecture, hyperparameters, or training procedure -- this exists because
we're training this substituted DenseCap component ourselves (the author's pretrained
checkpoint turned out to be unreachable via both OneDrive and BaiduYun), and a
multi-hour Colab job with zero resume support is too fragile to risk as-is:

1. Replaced NVIDIA Apex (`from apex import amp`) with native `torch.cuda.amp`
   (`autocast` + `GradScaler`). Apex predates PyTorch's own built-in mixed-precision
   support (added in torch 1.6) and requires a from-source CUDA-extension build that's
   a common source of exactly the kind of environment breakage we've been fighting all
   session -- native AMP does the same job (loss scaling + fp16 forward/backward) and
   ships with every modern PyTorch install.

2. Added checkpoint-resume support. Upstream `train.py` has NONE: it only saves when
   validation mAP improves at a 20k-iteration checkpoint, or once at the very end of
   all epochs -- meaning a Colab disconnect mid-run (very possible for a job this long)
   could lose everything with no way to continue. This version saves an unconditional
   "resume" checkpoint at the end of every epoch, and auto-loads it on startup if
   present, picking up at the next epoch.

Also wraps the `fasterrcnn_resnet50_fpn(pretrained=True)` call in a fallback for
newer torchvision, where the `pretrained=` kwarg has been replaced by `weights=`.

Everything else -- model config, loss weights, learning rates, batch size, epoch
count, optimizer, dataset construction -- is untouched from upstream.

Run this from inside the cloned densecap-pytorch repo (it imports `utils.data_loader`,
`model.densecap`, `evaluate` exactly like the original, so it needs that repo's other
files on the path):

    cd /content/densecap-pytorch
    cp /content/grounded-vqg-reproduction/scripts/densecap_train_patched.py .
    mkdir -p model_params
    python densecap_train_patched.py
"""
import os
import json

import torch
import numpy as np
from torch.utils.data.dataset import Subset
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from utils.data_loader import DenseCapDataset, DataLoaderPFG
from model.densecap import densecap_resnet50_fpn

from evaluate import quantity_check

torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 10  # paper-adjacent value from upstream's default -- unchanged
USE_TB = True

# CONFIG_PATH is the one path that should live on Drive, not local /content disk: it's
# where checkpoints get saved, and /content is wiped on every Colab disconnect --
# without this, the resume support above would have nothing to resume FROM after the
# exact kind of interruption it exists to survive. It's fine performance-wise on Drive
# because it's one moderately-large file written once per epoch, not the many-small-files
# pattern that made the COCO/Drive extraction so slow earlier.
#
# IMG_DIR_ROOT / VG_DATA_PATH / LOOK_UP_TABLES_PATH must stay on LOCAL disk -- these are
# read constantly during training (once per batch), and Visual Genome is ~108k small
# image files, so putting these on Drive would hit that exact same slow-FUSE problem.
CONFIG_PATH = os.environ.get('DENSECAP_MODEL_PARAMS_DIR', './model_params')
MODEL_NAME = 'train_all_val_all_bz_2_epoch_10_inject_init'
IMG_DIR_ROOT = os.environ.get('DENSECAP_IMG_DIR_ROOT', './data/visual-genome')
VG_DATA_PATH = os.environ.get('DENSECAP_VG_DATA_PATH', './data/VG-regions-lite.h5')
LOOK_UP_TABLES_PATH = os.environ.get('DENSECAP_LUT_PATH', './data/VG-regions-dicts-lite.pkl')
MAX_TRAIN_IMAGE = -1
MAX_VAL_IMAGE = -1

os.makedirs(CONFIG_PATH, exist_ok=True)
RESUME_PATH = os.path.join(CONFIG_PATH, '{}_resume.pth.tar'.format(MODEL_NAME))


def set_args():
    args = dict()
    args['backbone_pretrained'] = True
    args['return_features'] = False

    args['feat_size'] = 4096
    args['hidden_size'] = 512
    args['max_len'] = 16
    args['emb_size'] = 512
    args['rnn_num_layers'] = 1
    args['vocab_size'] = 10629
    args['fusion_type'] = 'init_inject'

    args['detect_loss_weight'] = 1.
    args['caption_loss_weight'] = 1.
    args['lr'] = 1e-4
    args['caption_lr'] = 1e-3
    args['weight_decay'] = 0.
    args['batch_size'] = 4
    args['use_pretrain_fasterrcnn'] = True
    args['box_detections_per_img'] = 50

    if not os.path.exists(os.path.join(CONFIG_PATH, MODEL_NAME)):
        os.mkdir(os.path.join(CONFIG_PATH, MODEL_NAME))
    with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'w') as f:
        json.dump(args, f, indent=2)

    return args


def load_fasterrcnn_pretrained():
    """torchvision >= 0.13 deprecated pretrained=True in favor of weights=; try the
    old kwarg first (matches upstream exactly on older torchvision) and fall back to
    the new one so this doesn't hard-crash on Colab's current torchvision version."""
    try:
        return fasterrcnn_resnet50_fpn(pretrained=True)
    except TypeError:
        return fasterrcnn_resnet50_fpn(weights="DEFAULT")


def save_model(model, optimizer, scaler, results_on_val, epoch, iter_counter, best_map, flag=None):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'results_on_val': results_on_val,
        'epoch': epoch,
        'iterations': iter_counter,
        'best_map': best_map,
    }
    if isinstance(flag, str):
        filename = os.path.join(CONFIG_PATH, '{}_{}.pth.tar'.format(MODEL_NAME, flag))
    else:
        filename = os.path.join(CONFIG_PATH, '{}.pth.tar'.format(MODEL_NAME))
    print('Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)


def train(args):
    print('Model {} start training...'.format(MODEL_NAME))

    model = densecap_resnet50_fpn(backbone_pretrained=args['backbone_pretrained'],
                                   feat_size=args['feat_size'],
                                   hidden_size=args['hidden_size'],
                                   max_len=args['max_len'],
                                   emb_size=args['emb_size'],
                                   rnn_num_layers=args['rnn_num_layers'],
                                   vocab_size=args['vocab_size'],
                                   fusion_type=args['fusion_type'],
                                   box_detections_per_img=args['box_detections_per_img'])
    if args['use_pretrain_fasterrcnn']:
        pretrained = load_fasterrcnn_pretrained()
        model.backbone.load_state_dict(pretrained.backbone.state_dict(), strict=False)
        model.rpn.load_state_dict(pretrained.rpn.state_dict(), strict=False)

    model.to(device)

    optimizer = torch.optim.Adam([{'params': (para for name, para in model.named_parameters()
                                    if para.requires_grad and 'box_describer' not in name)},
                                  {'params': (para for para in model.roi_heads.box_describer.parameters()
                                              if para.requires_grad), 'lr': args['caption_lr']}],
                                  lr=args['lr'], weight_decay=args['weight_decay'])

    scaler = GradScaler()

    start_epoch = 0
    iter_counter = 0
    best_map = 0.
    results = {}

    if os.path.exists(RESUME_PATH):
        print('Found resume checkpoint at {}, loading...'.format(RESUME_PATH))
        ckpt = torch.load(RESUME_PATH, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        iter_counter = ckpt['iterations']
        best_map = ckpt['best_map']
        results = ckpt.get('results_on_val', {})
        print('Resuming from epoch {} (iter {}, best_map {:.4f})'.format(start_epoch, iter_counter, best_map))

    train_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='train')
    val_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='val')
    idx_to_token = train_set.look_up_tables['idx_to_token']

    if MAX_TRAIN_IMAGE > 0:
        train_set = Subset(train_set, range(MAX_TRAIN_IMAGE))
    if MAX_VAL_IMAGE > 0:
        val_set = Subset(val_set, range(MAX_VAL_IMAGE))

    train_loader = DataLoaderPFG(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=2,
                                  pin_memory=True, collate_fn=DenseCapDataset.collate_fn)

    if USE_TB:
        writer = SummaryWriter()

    if start_epoch >= MAX_EPOCHS:
        print('Resume checkpoint already reached MAX_EPOCHS={}, nothing to do.'.format(MAX_EPOCHS))
        return

    for epoch in range(start_epoch, MAX_EPOCHS):

        for batch, (img, targets, info) in enumerate(train_loader):

            img = [img_tensor.to(device) for img_tensor in img]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            model.train()
            with autocast():
                losses = model(img, targets)

                detect_loss = losses['loss_objectness'] + losses['loss_rpn_box_reg'] + \
                               losses['loss_classifier'] + losses['loss_box_reg']
                caption_loss = losses['loss_caption']
                total_loss = args['detect_loss_weight'] * detect_loss + args['caption_loss_weight'] * caption_loss

            if USE_TB:
                writer.add_scalar('batch_loss/total', total_loss.item(), iter_counter)
                writer.add_scalar('batch_loss/detect_loss', detect_loss.item(), iter_counter)
                writer.add_scalar('batch_loss/caption_loss', caption_loss.item(), iter_counter)
                writer.add_scalar('details/loss_objectness', losses['loss_objectness'].item(), iter_counter)
                writer.add_scalar('details/loss_rpn_box_reg', losses['loss_rpn_box_reg'].item(), iter_counter)
                writer.add_scalar('details/loss_classifier', losses['loss_classifier'].item(), iter_counter)
                writer.add_scalar('details/loss_box_reg', losses['loss_box_reg'].item(), iter_counter)

            if iter_counter % (len(train_set) / (args['batch_size'] * 16)) == 0:
                print("[{}][{}]\ntotal_loss {:.3f}".format(epoch, batch, total_loss.item()))
                for k, v in losses.items():
                    print(" <{}> {:.3f}".format(k, v))

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if iter_counter > 0 and iter_counter % 20000 == 0:
                try:
                    results = quantity_check(model, val_set, idx_to_token, device, max_iter=-1, verbose=True)
                    if results['map'] > best_map:
                        best_map = results['map']
                        # flag='best' -- must NOT be the bare {MODEL_NAME}.pth.tar filename,
                        # since the end-of-epoch resume save below writes there too and
                        # renames it to RESUME_PATH; without a distinct flag the two
                        # checkpoints collide and this "best" snapshot gets silently
                        # overwritten/renamed away the moment the current epoch ends.
                        save_model(model, optimizer, scaler, results, epoch, iter_counter, best_map, flag='best')

                    if USE_TB:
                        writer.add_scalar('metric/map', results['map'], iter_counter)
                        writer.add_scalar('metric/det_map', results['detmap'], iter_counter)

                except AssertionError as e:
                    print('[INFO]: evaluation failed at epoch {}'.format(epoch))
                    print(e)

            iter_counter += 1

        # unconditional end-of-epoch checkpoint -- this is the resume safety net
        # upstream doesn't have; overwritten every epoch, not kept as history.
        save_model(model, optimizer, scaler, results, epoch, iter_counter, best_map)
        os.replace(
            os.path.join(CONFIG_PATH, '{}.pth.tar'.format(MODEL_NAME)),
            RESUME_PATH,
        )
        print('Epoch {} complete, resume checkpoint updated.'.format(epoch))

    save_model(model, optimizer, scaler, results, MAX_EPOCHS - 1, iter_counter, best_map, flag='end')

    if USE_TB:
        writer.close()


if __name__ == '__main__':
    args = set_args()
    train(args)
