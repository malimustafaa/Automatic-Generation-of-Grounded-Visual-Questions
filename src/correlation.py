"""Caption Encoder + Image/Caption Correlation module -- paper Sec 3.2, 'Generate Questions'.

Encoder:
  'we found it useful to let the LSTM encoder ... read the image features prior to
  reading captions. In particular, at time step t=0, we initialize the state vector
  m_0 to zero and feed the image features as x_0. At the 1st time step, the encoder
  reads in a special token S0 ... After reading the whole caption of length L, the
  encoder yields the last state vector m_L as the embedding of caption.'

  NOTE: this module feeds [image_feature, caption_words] (paper's x_0 is the image
  feature; the S0 start-token step is folded in by the caller via the caption's own
  leading <start> token during tokenization -- see src/dataset.py). The output m_L
  (here: the LSTM's final hidden state) is the caption embedding.

Correlation:
  'The correlation module takes as input the caption embeddings from the encoder and
  the image features from VGG-16, produces a 300-dimensional joint feature map. We
  apply a linear layer of size 300 x 600 and a PReLU layer in sequel.'

  A 300x600 linear layer producing a 300-d *output* from two concatenated 300-d
  *inputs* (600 total) is Linear(in_features=600, out_features=300) -- so the caption
  encoder's hidden size is PINNED to 300 here (not a free/gap-filled choice like the
  other LSTMs in this codebase), because that's the only value consistent with the
  paper's explicitly stated 600-dim linear layer input.
"""
import torch
import torch.nn as nn

CAPTION_ENCODER_HIDDEN_SIZE = 300  # pinned by the paper's stated 300x600 correlation linear layer


class CaptionEncoder(nn.Module):
    def __init__(self, word_emb_dim: int = 300, hidden_size: int = CAPTION_ENCODER_HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(word_emb_dim, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, image_feat: torch.Tensor, caption_embeds: torch.Tensor,
                caption_lengths: torch.Tensor) -> torch.Tensor:
        # image_feat: (B, 300) fed as x_0; caption_embeds: (B, L, 300) fed as x_1..x_L
        x0 = image_feat.unsqueeze(1)
        seq = torch.cat([x0, caption_embeds], dim=1)  # (B, 1+L, 300)
        lengths = caption_lengths + 1
        packed = nn.utils.rnn.pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return h_n[-1]  # (B, hidden_size) -- "caption embedding" m_L


class CorrelationLayer(nn.Module):
    def __init__(self, caption_emb_dim: int = CAPTION_ENCODER_HIDDEN_SIZE,
                 image_feat_dim: int = 300, joint_dim: int = 300):
        super().__init__()
        self.linear = nn.Linear(caption_emb_dim + image_feat_dim, joint_dim)
        self.prelu = nn.PReLU()

    def forward(self, caption_embedding: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([caption_embedding, image_feat], dim=-1)
        return self.prelu(self.linear(joint))  # (B, 300)
