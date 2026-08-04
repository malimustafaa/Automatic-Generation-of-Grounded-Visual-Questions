"""P(t_n | c_n) -- Sec 3.1:

'The first one is a Long Short Term Memory (LSTM) that maps a caption into a hidden
representation... As the representation of the whole sequence, we take the last state
h_L... This representation is further fed into a softmax layer to compute a probability
vector for all question types.'

Six types, in the paper's stated order: what, when, where, who, why, how.
LSTM hidden size is not given by the paper -- gap-filled (see configs/default.yaml).
"""
import torch
import torch.nn as nn

QUESTION_TYPES = ["what", "when", "where", "who", "why", "how"]


class QuestionTypeSelector(nn.Module):
    def __init__(self, word_emb_dim: int = 300, hidden_size: int = 512, num_types: int = len(QUESTION_TYPES)):
        super().__init__()
        self.lstm = nn.LSTM(word_emb_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_types)

    def forward(self, caption_embeds: torch.Tensor, caption_lengths: torch.Tensor) -> torch.Tensor:
        # caption_embeds: (B, L, 300) -> logits: (B, num_types)
        packed = nn.utils.rnn.pack_padded_sequence(
            caption_embeds, caption_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        return self.classifier(h_n[-1])
