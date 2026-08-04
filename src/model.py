"""GroundedVQGModel: wires the type selector, caption encoder + correlation layer,
and decoder together, and implements the per-instance losses that src/train.py
combines with the EM latent-caption weight (Eq. 6) into the paper's full objective.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .correlation import CAPTION_ENCODER_HIDDEN_SIZE, CaptionEncoder, CorrelationLayer
from .decoder import QuestionDecoder
from .question_type_selector import QuestionTypeSelector


class GroundedVQGModel(nn.Module):
    def __init__(self, embedding: nn.Embedding, vocab_size: int,
                 type_hidden: int = 512, decoder_hidden: int = 512):
        super().__init__()
        self.embedding = embedding
        self.type_selector = QuestionTypeSelector(hidden_size=type_hidden)
        self.encoder = CaptionEncoder(hidden_size=CAPTION_ENCODER_HIDDEN_SIZE)
        self.correlation = CorrelationLayer(caption_emb_dim=CAPTION_ENCODER_HIDDEN_SIZE,
                                             image_feat_dim=300, joint_dim=300)
        self.decoder = QuestionDecoder(hidden_size=decoder_hidden, vocab_size=vocab_size)

    def forward_step(self, image_feat: torch.Tensor, caption_ids: torch.Tensor,
                      caption_lengths: torch.Tensor, decoder_input_ids: torch.Tensor,
                      decoder_lengths: torch.Tensor, start_id: int):
        caption_embeds = self.embedding(caption_ids)
        type_logits = self.type_selector(caption_embeds, caption_lengths)

        caption_embedding = self.encoder(image_feat, caption_embeds, caption_lengths)
        joint_feature = self.correlation(caption_embedding, image_feat)

        start_embed = self.embedding(
            torch.full((image_feat.size(0),), start_id, dtype=torch.long, device=image_feat.device))
        target_embeds = self.embedding(decoder_input_ids)
        logits = self.decoder.forward_train(joint_feature, start_embed, target_embeds, decoder_lengths)
        return type_logits, logits

    @staticmethod
    def per_instance_loss(type_logits: torch.Tensor, type_target: torch.Tensor,
                           question_logits: torch.Tensor, question_target: torch.Tensor,
                           question_lengths: torch.Tensor, pad_id: int):
        """Returns (type_loss, q_loss), both shape (B,) -- un-reduced per-instance NLL,
        matching Eq. 6's -log[P(q|c,x,t) * P(t|c)] = -log P(q|...) - log P(t|c)."""
        type_loss = F.cross_entropy(type_logits, type_target, reduction="none")

        B, T, V = question_logits.shape
        q_loss = F.cross_entropy(
            question_logits.reshape(-1, V), question_target.reshape(-1),
            ignore_index=pad_id, reduction="none",
        ).view(B, T).sum(dim=1) / (question_lengths.clamp(min=1).float() + 1)  # +1 for the <end> token

        return type_loss, q_loss
