"""LSTM decoder + softmax -- paper Sec 3.2:

'Our decoder extends the LSTM decoder of [Vinyals et al., 2015] with a ngram language
model. The LSTM decoder consists of an LSTM layer and a softmax layer. The LSTM layer
starts with reading the joint feature map and the start token S0 in the same fashion
as the caption encoder. From time step t=0, the softmax layer predicts the most likely
word given the state vector at time t yielded by the LSTM layer.'

Hidden size is not given by the paper -- gap-filled (see configs/default.yaml).
The n-gram (Kneser-Ney bigram) interpolation described in the same section lives in
src/bigram_lm.py and is applied at inference time in src/generate.py, exactly as the
paper's 'Joint decoding' paragraph specifies.
"""
import torch
import torch.nn as nn


class QuestionDecoder(nn.Module):
    def __init__(self, word_emb_dim: int = 300, joint_dim: int = 300,
                 hidden_size: int = 512, vocab_size: int = None):
        super().__init__()
        assert vocab_size is not None
        self.lstm = nn.LSTM(word_emb_dim, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward_train(self, joint_feature: torch.Tensor, start_embed: torch.Tensor,
                       target_embeds: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """Teacher-forced training pass.
        joint_feature fed as x_0 ('reading the joint feature map ... in the same fashion
        as the caption encoder'), then <start>, then the target words.
        Returns logits aligned with [<start>, w_1, ..., w_L] (i.e. predicting w_1..w_L, <end>).
        """
        x0 = joint_feature.unsqueeze(1)
        seq = torch.cat([x0, start_embed.unsqueeze(1), target_embeds], dim=1)
        lengths = target_lengths + 2
        packed = nn.utils.rnn.pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        logits = self.output(out[:, 1:, :])  # drop the joint-feature timestep's own output
        return logits  # (B, 1+L, vocab_size)

    def init_state(self, joint_feature: torch.Tensor, start_embed: torch.Tensor):
        """Run the joint-feature + <start> steps once, return the hidden state ready
        for autoregressive stepping (used at inference time by src/generate.py)."""
        x0 = joint_feature.unsqueeze(1)
        _, hidden = self.lstm(x0)
        out, hidden = self.lstm(start_embed.unsqueeze(1), hidden)
        logits = self.output(out.squeeze(1))
        return logits, hidden

    def step(self, x_t: torch.Tensor, hidden):
        out, hidden = self.lstm(x_t, hidden)
        logits = self.output(out.squeeze(1))
        return logits, hidden
