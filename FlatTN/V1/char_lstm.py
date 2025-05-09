"""
Character LSTM implementation (matches https://arxiv.org/pdf/1805.01052.pdf)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharacterLSTM(nn.Module):
    # def __init__(self, num_embeddings, d_embedding, d_out, char_dropout=0.2, **kwargs):
    #     super().__init__()
    def __init__(self, bert_embedding, d_embedding, d_out, char_dropout=0.2, **kwargs):
        super().__init__()

        self.d_embedding = d_embedding
        self.d_out = d_out

        self.lstm = nn.LSTM(
            self.d_embedding, self.d_out // 2, num_layers=1, bidirectional=True
        )
        self.emb = bert_embedding
        # self.emb = nn.Embedding(num_embeddings, self.d_embedding, **kwargs)
        self.char_dropout = nn.Dropout(char_dropout)

    def forward(self, chars_packed, valid_token_mask):
        inp_embs = nn.utils.rnn.PackedSequence(
            self.char_dropout(self.emb(chars_packed.data)),
            batch_sizes=chars_packed.batch_sizes,
            sorted_indices=chars_packed.sorted_indices,
            unsorted_indices=chars_packed.unsorted_indices,
        )

        _, (lstm_out, _) = self.lstm(inp_embs)
        lstm_out = torch.cat([lstm_out[0], lstm_out[1]], -1)

        # Switch to a representation where there are dummy vectors for invalid
        # tokens generated by padding.
        res = lstm_out.new_zeros(
            (valid_token_mask.shape[0], valid_token_mask.shape[1], lstm_out.shape[-1])
        )
        res[valid_token_mask] = lstm_out
        return res


class RetokenizerForCharLSTM:
    # Assumes that these control characters are not present in treebank text
    CHAR_UNK = "\0"
    CHAR_ID_UNK = 0
    CHAR_START_SENTENCE = "\1"
    CHAR_START_WORD = "\2"
    CHAR_STOP_WORD = "\3"
    CHAR_STOP_SENTENCE = "\4"

    def __init__(self, char_vocab):
        self.char_vocab = char_vocab

    @classmethod
    def build_vocab(cls, sentences):
        char_set = set()
        for sentence in sentences:
            if isinstance(sentence, tuple):
                sentence = sentence[0]
            for word in sentence:
                char_set |= set(word)

        # If codepoints are small (e.g. Latin alphabet), index by codepoint
        # directly
        highest_codepoint = max(ord(char) for char in char_set)
        if highest_codepoint < 512:
            if highest_codepoint < 256:
                highest_codepoint = 256
            else:
                highest_codepoint = 512

            char_vocab = {}
            # This also takes care of constants like CHAR_UNK, etc.
            for codepoint in range(highest_codepoint):
                char_vocab[chr(codepoint)] = codepoint
            return char_vocab
        else:
            char_vocab = {}
            char_vocab[cls.CHAR_UNK] = 0
            char_vocab[cls.CHAR_START_SENTENCE] = 1
            char_vocab[cls.CHAR_START_WORD] = 2
            char_vocab[cls.CHAR_STOP_WORD] = 3
            char_vocab[cls.CHAR_STOP_SENTENCE] = 4
            for id_, char in enumerate(sorted(char_set), start=5):
                char_vocab[char] = id_
            return char_vocab

    def __call__(self, words, space_after="ignored", return_tensors=None):
        if return_tensors != "np":
            raise NotImplementedError("Only return_tensors='np' is supported.")

        res = {}

        # Sentence-level start/stop tokens are encoded as 3 pseudo-chars
        # Within each word, account for 2 start/stop characters
        max_word_len = max(3, max(len(word) for word in words)) + 2
        char_ids = np.zeros((len(words) + 2, max_word_len), dtype=int)
        word_lens = np.zeros(len(words) + 2, dtype=int)

        char_ids[0, :5] = [
            self.char_vocab[self.CHAR_START_WORD],
            self.char_vocab[self.CHAR_START_SENTENCE],
            self.char_vocab[self.CHAR_START_SENTENCE],
            self.char_vocab[self.CHAR_START_SENTENCE],
            self.char_vocab[self.CHAR_STOP_WORD],
        ]
        word_lens[0] = 5
        for i, word in enumerate(words, start=1):
            char_ids[i, 0] = self.char_vocab[self.CHAR_START_WORD]
            for j, char in enumerate(word, start=1):
                char_ids[i, j] = self.char_vocab.get(char, self.CHAR_ID_UNK)
            char_ids[i, j + 1] = self.char_vocab[self.CHAR_STOP_WORD]
            word_lens[i] = j + 2
        char_ids[i + 1, :5] = [
            self.char_vocab[self.CHAR_START_WORD],
            self.char_vocab[self.CHAR_STOP_SENTENCE],
            self.char_vocab[self.CHAR_STOP_SENTENCE],
            self.char_vocab[self.CHAR_STOP_SENTENCE],
            self.char_vocab[self.CHAR_STOP_WORD],
        ]
        word_lens[i + 1] = 5

        res["char_ids"] = char_ids
        res["word_lens"] = word_lens
        res["valid_token_mask"] = np.ones_like(word_lens, dtype=bool)

        return res

    def pad(self, examples, return_tensors=None):
        if return_tensors != "pt":
            raise NotImplementedError("Only return_tensors='pt' is supported.")
        max_word_len = max(example["char_ids"].shape[-1] for example in examples)
        char_ids = torch.cat(
            [
                F.pad(
                    torch.tensor(example["char_ids"]),
                    (0, max_word_len - example["char_ids"].shape[-1]),
                )
                for example in examples
            ]
        )
        word_lens = torch.cat(
            [torch.tensor(example["word_lens"]) for example in examples]
        )
        valid_token_mask = nn.utils.rnn.pad_sequence(
            [torch.tensor(example["valid_token_mask"]) for example in examples],
            batch_first=True,
            padding_value=False,
        )

        char_ids = nn.utils.rnn.pack_padded_sequence(
            char_ids, word_lens, batch_first=True, enforce_sorted=False
        )
        return {
            "char_ids": char_ids,
            "valid_token_mask": valid_token_mask,
        }