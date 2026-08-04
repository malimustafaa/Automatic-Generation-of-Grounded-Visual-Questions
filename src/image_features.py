"""Frozen VGG-16 image feature extractor (paper Sec 3.2 / 4.4):

'we apply a VGG-16 model trained on ImageNet without fine-tuning to produce
300-dimensional feature vectors. The dimension is chosen to match the size of the
pre-trained word embeddings.'

VGG-16's own fc7 layer outputs 4096-d, not 300-d. The paper doesn't detail the
projection from 4096 -> 300, so the Linear(4096, 300) below is a documented gap-fill --
it preserves the paper's *stated interface* (a 300-d image vector matching GloVe's
dimensionality) without altering VGG-16 itself, which stays frozen exactly as specified.
"""
import torch
import torch.nn as nn
import torchvision


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, out_dim: int = 300):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        # up to and including fc7 + ReLU + Dropout, dropping only the final 1000-way classifier head
        self.fc6_fc7 = nn.Sequential(*list(vgg.classifier.children())[:-1])
        for p in self.parameters():
            p.requires_grad = False  # frozen, no fine-tuning -- paper Sec 3.2/4.4
        self.eval()
        self.project = nn.Linear(4096, out_dim)  # trainable; VGG-16 itself stays frozen

    @torch.no_grad()
    def _vgg_forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc6_fc7(x)
        return x  # (B, 4096)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self._vgg_forward(images)
        return self.project(feat)  # (B, out_dim)
