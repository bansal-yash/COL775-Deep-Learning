"""
tokenizer.py - Simple word-level tokenizer built from CLEVR training captions.

CLEVR captions have the structured form:
  "An image with N objects: 1 small red metal cube, 2 large blue rubber spheres"

The vocabulary is constructed directly from the training captions (no pretrained
tokenizer) as required by the assignment.
"""

import json
import re
from collections import Counter
from typing import List, Tuple


class CLEVRTokenizer:
    """
    Word-level tokenizer for CLEVR captions.

    Special tokens
    ──────────────
      <pad>  — padding token (id=0)
      <sos>  — start-of-sequence (id=1)
      <eos>  — end-of-sequence   (id=2)
      <unk>  — unknown word      (id=3)

    Usage
    ─────
      tok = CLEVRTokenizer()
      tok.build_vocab(all_captions)
      ids, mask = tok.encode("An image with 3 objects: 1 large red cube", max_len=77)
    """

    PAD = '<pad>'
    SOS = '<sos>'
    EOS = '<eos>'
    UNK = '<unk>'

    def __init__(self):
        self.word2idx: dict = {}
        self.idx2word: dict = {}
        self.vocab_size: int = 0
        self.pad_id: int = 0
        self.sos_id: int = 1
        self.eos_id: int = 2
        self.unk_id: int = 3

    # ─────────────────────────────────────── normalisation helper ────────── #
    @staticmethod
    def _normalise(caption: str) -> str:
        """Lower-case and separate punctuation so ':' and ',' become tokens."""
        caption = caption.lower()
        caption = caption.replace(':', ' : ').replace(',', ' ,')
        # Collapse multiple spaces
        caption = re.sub(r'\s+', ' ', caption).strip()
        return caption

    # ─────────────────────────────────────── build vocabulary ────────────── #
    def build_vocab(self, captions: List[str], min_freq: int = 1) -> 'CLEVRTokenizer':
        """
        Build the vocabulary from a list of caption strings.

        Args:
            captions:  list of raw caption strings
            min_freq:  minimum occurrence count to include a word

        Returns self (for chaining).
        """
        counts: Counter = Counter()
        for cap in captions:
            counts.update(self._normalise(cap).split())

        # Words that meet the frequency threshold, sorted for determinism
        words = sorted(w for w, c in counts.items() if c >= min_freq)

        vocab = [self.PAD, self.SOS, self.EOS, self.UNK] + words
        self.word2idx  = {w: i for i, w in enumerate(vocab)}
        self.idx2word  = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)

        self.pad_id = self.word2idx[self.PAD]
        self.sos_id = self.word2idx[self.SOS]
        self.eos_id = self.word2idx[self.EOS]
        self.unk_id = self.word2idx[self.UNK]
        return self

    # ─────────────────────────────────────── encode / decode ─────────────── #
    def encode(self, caption: str, max_len: int = 77) -> Tuple[List[int], List[int]]:
        """
        Encode one caption to (token_ids, attention_mask), both of length max_len.

        The sequence is:
          [SOS] w1 w2 ... wN [EOS] [PAD] [PAD] ...

        Returns:
            token_ids      – list of int, length max_len
            attention_mask – list of int (1=real, 0=pad), length max_len
        """
        words = self._normalise(caption).split()
        ids = ([self.sos_id]
               + [self.word2idx.get(w, self.unk_id) for w in words]
               + [self.eos_id])

        # Truncate if too long (keep EOS)
        if len(ids) > max_len:
            ids = ids[:max_len - 1] + [self.eos_id]

        real_len  = len(ids)
        pad_len   = max_len - real_len
        attn_mask = [1] * real_len + [0] * pad_len
        ids       = ids + [self.pad_id] * pad_len

        return ids, attn_mask

    def decode(self, token_ids: List[int]) -> str:
        """Convert token ids back to a readable string (no special tokens)."""
        skip = {self.pad_id, self.sos_id, self.eos_id}
        words = [self.idx2word.get(i, self.UNK) for i in token_ids if i not in skip]
        return ' '.join(words)

    # ─────────────────────────────────────── persistence ─────────────────── #
    def save(self, path: str):
        data = {
            'word2idx':  self.word2idx,
            'idx2word':  {str(k): v for k, v in self.idx2word.items()},
            'vocab_size': self.vocab_size,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'CLEVRTokenizer':
        tok = cls()
        with open(path) as f:
            data = json.load(f)
        tok.word2idx  = data['word2idx']
        tok.idx2word  = {int(k): v for k, v in data['idx2word'].items()}
        tok.vocab_size = data['vocab_size']
        tok.pad_id = tok.word2idx[cls.PAD]
        tok.sos_id = tok.word2idx[cls.SOS]
        tok.eos_id = tok.word2idx[cls.EOS]
        tok.unk_id = tok.word2idx[cls.UNK]
        return tok
