"""EXPERIMENTAL, NOT part of the paper-faithful reproduction: adds Bahdanau-style
attention so the question decoder can look back at individual caption words at every
decoding step, instead of only ever seeing the single fixed-size joint feature vector
the paper's own architecture produces once, at the very start of decoding (see
src/correlation.py / src/decoder.py -- both untouched by this file, and by everything
else in this module).

Why: the paper's own architecture (Sec 3.2) compresses the whole caption into one
300-d vector before the decoder starts generating, and the decoder never looks back at
it again. On an undertrained checkpoint (what this reproduction's compute/data budget
actually produces), the decoder tends to fall back on frequent, caption-independent
completions rather than using that compressed signal (see the cell 10 demo discussion
in notebooks/colab_train.ipynb). Attention is the standard fix for exactly this kind of
encoder bottleneck -- but it's a real architecture change beyond what the paper
specifies, so it lives entirely in this separate module/checkpoints/notebook cells,
never touching model.py, decoder.py, correlation.py, or the paper-faithful training run
(src/train.py is not imported for its model, only for its generic, model-agnostic
helpers).

Kept lightweight on purpose, in the same spirit as the paper's own design: attention
here is only over the caption's own tokens (a handful of words, typically <20), not
over image regions/pixels -- the expensive kind of attention this paper's "lightweight"
framing was explicitly avoiding. See scripts/train_attention_experiment.py, which
prints a parameter-count comparison against the paper-faithful model before any real
training run.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .correlation import CAPTION_ENCODER_HIDDEN_SIZE, CorrelationLayer
from .model import GroundedVQGModel
from .question_type_selector import QuestionTypeSelector


class AttentionCaptionEncoder(nn.Module):
    """Same [image_feat, caption_words] input as src/correlation.py's CaptionEncoder,
    but returns the per-timestep outputs (not just the final hidden state) so the
    decoder's attention has something to attend over."""
    def __init__(self, word_emb_dim: int = 300, hidden_size: int = CAPTION_ENCODER_HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(word_emb_dim, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, image_feat: torch.Tensor, caption_embeds: torch.Tensor, caption_lengths: torch.Tensor):
        x0 = image_feat.unsqueeze(1)
        seq = torch.cat([x0, caption_embeds], dim=1)  # (B, 1+L, 300)
        lengths = caption_lengths + 1
        packed = nn.utils.rnn.pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return outputs, h_n[-1]  # (B, 1+L, hidden) for attention; (B, hidden) kept for parity with CaptionEncoder


class BahdanauAttention(nn.Module):
    def __init__(self, decoder_hidden: int, encoder_hidden: int, attn_dim: int = 256):
        super().__init__()
        self.w_dec = nn.Linear(decoder_hidden, attn_dim)
        self.w_enc = nn.Linear(encoder_hidden, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor):
        # decoder_hidden: (B, decoder_hidden); encoder_outputs: (B, L, encoder_hidden); mask: (B, L) bool
        scores = self.v(torch.tanh(
            self.w_dec(decoder_hidden).unsqueeze(1) + self.w_enc(encoder_outputs)
        )).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, L)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, encoder_hidden)
        return context, weights


class AttentionQuestionDecoder(nn.Module):
    """Same role as src/decoder.py's QuestionDecoder, but built on an LSTMCell run one
    step at a time (including during teacher-forced training) so attention can be
    recomputed at every step -- the packed/batched nn.LSTM the paper-faithful decoder
    uses can't do that, since attention weights depend on the decoder's own hidden
    state at each individual timestep, not just the final one."""
    def __init__(self, word_emb_dim: int = 300, hidden_size: int = 512, vocab_size: int = None,
                 encoder_hidden: int = CAPTION_ENCODER_HIDDEN_SIZE, attn_dim: int = 256):
        super().__init__()
        assert vocab_size is not None
        self.hidden_size = hidden_size
        self.encoder_hidden = encoder_hidden
        self.lstm_cell = nn.LSTMCell(word_emb_dim + encoder_hidden, hidden_size)
        self.attention = BahdanauAttention(hidden_size, encoder_hidden, attn_dim)
        self.output = nn.Linear(hidden_size + encoder_hidden, vocab_size)

    def init_state(self, joint_feature: torch.Tensor):
        """Reads the joint feature map as step 0, same convention as
        QuestionDecoder.init_state -- fed with a zero attention context since there's
        no decoder hidden state yet to attend with."""
        B, device = joint_feature.size(0), joint_feature.device
        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)
        zero_ctx = torch.zeros(B, self.encoder_hidden, device=device)
        return self.lstm_cell(torch.cat([joint_feature, zero_ctx], dim=-1), (h, c))

    def step(self, x_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor,
             encoder_outputs: torch.Tensor, mask: torch.Tensor):
        """x_t, h, c: (B, dim) -- 2D, unlike QuestionDecoder.step's (B, 1, dim), since
        LSTMCell (not batched nn.LSTM) operates on non-sequence tensors."""
        context, weights = self.attention(h, encoder_outputs, mask)
        h, c = self.lstm_cell(torch.cat([x_t, context], dim=-1), (h, c))
        logits = self.output(torch.cat([h, context], dim=-1))
        return logits, h, c, weights

    def forward_train(self, joint_feature: torch.Tensor, start_embed: torch.Tensor,
                       target_embeds: torch.Tensor, target_lengths: torch.Tensor,
                       encoder_outputs: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
        """target_lengths is accepted but unused (kept only so this has the same call
        signature as QuestionDecoder.forward_train) -- this unrolled version runs every
        sample in the batch for the same number of steps regardless of its real length,
        relying on per_instance_loss's ignore_index=pad_id to mask out the padded
        positions, rather than on pack_padded_sequence trimming them beforehand."""
        h, c = self.init_state(joint_feature)
        inputs = torch.cat([start_embed.unsqueeze(1), target_embeds], dim=1)  # [start, w_1..w_L]
        all_logits = []
        for t in range(inputs.size(1)):
            logits, h, c, _ = self.step(inputs[:, t, :], h, c, encoder_outputs, encoder_mask)
            all_logits.append(logits)
        return torch.stack(all_logits, dim=1)  # (B, L+1, vocab) predicting [w_1..w_L, <end>]


def build_encoder_mask(lengths: torch.Tensor, max_len: int, device) -> torch.Tensor:
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return idx < lengths.to(device).unsqueeze(1)


class AttentionGroundedVQGModel(nn.Module):
    """Drop-in analogue of src/model.py's GroundedVQGModel -- same forward_step
    signature, and per_instance_loss is literally the same function (it only depends
    on logits/target shapes, not which decoder produced them). This lets
    scripts/train_attention_experiment.py reuse src/train.py's generic, model-agnostic
    helpers (compute_em_weights, build_vocab_and_idf) without modifying src/train.py at
    all. type_selector and correlation are unchanged from the paper-faithful model --
    only the encoder/decoder pair is swapped for the attention-augmented versions above.
    """
    def __init__(self, embedding: nn.Embedding, vocab_size: int,
                 type_hidden: int = 512, decoder_hidden: int = 512, attn_dim: int = 256):
        super().__init__()
        self.embedding = embedding
        self.type_selector = QuestionTypeSelector(hidden_size=type_hidden)
        self.encoder = AttentionCaptionEncoder(hidden_size=CAPTION_ENCODER_HIDDEN_SIZE)
        self.correlation = CorrelationLayer(caption_emb_dim=CAPTION_ENCODER_HIDDEN_SIZE,
                                             image_feat_dim=300, joint_dim=300)
        self.decoder = AttentionQuestionDecoder(hidden_size=decoder_hidden, vocab_size=vocab_size,
                                                 encoder_hidden=CAPTION_ENCODER_HIDDEN_SIZE, attn_dim=attn_dim)

    def forward_step(self, image_feat: torch.Tensor, caption_ids: torch.Tensor,
                      caption_lengths: torch.Tensor, decoder_input_ids: torch.Tensor,
                      decoder_lengths: torch.Tensor, start_id: int):
        caption_embeds = self.embedding(caption_ids)
        type_logits = self.type_selector(caption_embeds, caption_lengths)

        encoder_outputs, caption_embedding = self.encoder(image_feat, caption_embeds, caption_lengths)
        joint_feature = self.correlation(caption_embedding, image_feat)
        encoder_mask = build_encoder_mask(caption_lengths + 1, encoder_outputs.size(1), image_feat.device)

        start_embed = self.embedding(
            torch.full((image_feat.size(0),), start_id, dtype=torch.long, device=image_feat.device))
        target_embeds = self.embedding(decoder_input_ids)
        logits = self.decoder.forward_train(joint_feature, start_embed, target_embeds, decoder_lengths,
                                             encoder_outputs, encoder_mask)
        return type_logits, logits

    per_instance_loss = staticmethod(GroundedVQGModel.per_instance_loss)


@torch.no_grad()
def generate_questions_attention(model: AttentionGroundedVQGModel, vocab, image_feat: torch.Tensor,
                                  candidates, num_questions: int = 6, max_len: int = 20,
                                  device: str = "cpu"):
    """Greedy-only, no bigram interpolation/beam search/dedup/distinct-types -- kept
    deliberately simple, since the point of this experiment is to check whether
    attention changes what the model grounds its output in at all, not to reproduce
    every inference-time trick already built for the paper-faithful decoder in
    src/generate.py. If attention shows a real improvement, those tricks can be ported
    over onto this decoder's step() the same way they were layered onto the original."""
    from .dataset import tokenize
    from .question_type_selector import QUESTION_TYPES
    from .vocab import END, START

    model.eval()
    image_feat_b = image_feat.to(device).unsqueeze(0)

    conf = torch.tensor([c.get("confidence", 1.0) for c in candidates], dtype=torch.float32)
    prior = conf / conf.sum()

    questions = []
    for _ in range(num_questions):
        cap_idx = int(torch.multinomial(prior, 1).item())
        caption = candidates[cap_idx]["caption"]
        caption_tokens = tokenize(caption)[:max_len]
        caption_ids = torch.tensor([vocab.encode(caption_tokens)], dtype=torch.long, device=device)
        caption_lengths = torch.tensor([max(len(caption_tokens), 1)])

        caption_embeds = model.embedding(caption_ids)
        type_logits = model.type_selector(caption_embeds, caption_lengths)
        type_probs = F.softmax(type_logits, dim=-1).squeeze(0)
        type_idx = int(torch.multinomial(type_probs, 1).item())
        qtype = QUESTION_TYPES[type_idx]

        encoder_outputs, caption_embedding = model.encoder(image_feat_b, caption_embeds, caption_lengths)
        joint_feature = model.correlation(caption_embedding, image_feat_b)
        encoder_mask = build_encoder_mask(caption_lengths + 1, encoder_outputs.size(1), device)

        h, c = model.decoder.init_state(joint_feature)
        start_embed = model.embedding(torch.tensor([vocab.word2idx[START]], device=device))
        _, h, c, _ = model.decoder.step(start_embed, h, c, encoder_outputs, encoder_mask)

        generated = [qtype]  # k=1 fixed opener, same convention as src/generate.py's TYPE_PREFIXES
        next_input_id = vocab.word2idx.get(qtype, vocab.word2idx["<unk>"])

        for _ in range(max_len - len(generated)):
            x_t = model.embedding(torch.tensor([next_input_id], device=device))
            logits, h, c, _ = model.decoder.step(x_t, h, c, encoder_outputs, encoder_mask)
            next_id = int(torch.argmax(logits.squeeze(0)).item())
            next_word = vocab.idx2word[next_id]
            if next_word == END:
                break
            generated.append(next_word)
            next_input_id = next_id

        text = " ".join(generated)
        questions.append(text if generated[-1] == "?" else text + " ?")

    return questions
