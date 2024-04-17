from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List, Optional
from megalodon.config import TokenizerConf
import os


logger = getLogger()


class Tokenizer:
    def __init__(self, tokenizer_cfg: TokenizerConf):
        # reload tokenizer
        model_path = tokenizer_cfg.tokenizer_path
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        self.additional_vocab_size = tokenizer_cfg.additional_vocab_size
        self.sp_model_vocab_size = self.sp_model.vocab_size()
        self.used_special_tokens = 0
        self.already_tokenized_data = tokenizer_cfg.data_tokenized

        assert tokenizer_cfg.additional_vocab_size % 256 == 0, \
            f"additional vocab size is not multiple of 256: {tokenizer_cfg.additional_vocab_size}"

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size() + tokenizer_cfg.additional_vocab_size
        self._bos_id: Optional[int] = None
        self._eos_id: Optional[int] = None
        self._pad_id: Optional[int] = None
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: [str, List[int]], bos: bool, eos: bool) -> List[int]:
        if self.already_tokenized_data:
            t = s
        else:
            assert type(s) is str
            t = self.sp_model.encode(s)
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def encode_str(self, s: str) -> List[str]:
        assert type(s) is str
        return self.sp_model.encode(s, out_type=str)

    def decode(self, tokens: List[int], cut_at_eos: bool = True) -> str:
        if cut_at_eos:
            for k, t in enumerate(tokens):
                if t == self.eos_id:
                    tokens = tokens[: k + 1]
                    break
        return self.sp_model.decode(tokens)

    @property
    def bos_id(self) -> int:
        if self._bos_id is not None:
            return self._bos_id

        if self.sp_model.bos_id() != -1 or self.additional_vocab_size == 0:
            self._bos_id = self.sp_model.bos_id()
            return self._bos_id

        bos_id = self.sp_model_vocab_size + self.used_special_tokens
        self.used_special_tokens += 1
        self._bos_id = bos_id
        return bos_id

    @property
    def eos_id(self) -> int:
        if self._eos_id is not None:
            return self._eos_id

        if self.sp_model.eos_id() != -1 or self.additional_vocab_size == 0:
            self._eos_id = self.sp_model.eos_id()
            return self._eos_id

        eos_id = self.sp_model_vocab_size + self.used_special_tokens
        self.used_special_tokens += 1
        self._eos_id = eos_id
        return eos_id

    @property
    def pad_id(self) -> int:
        if self._pad_id is not None:
            return self._pad_id

        if self.sp_model.pad_id() != -1 or self.additional_vocab_size == 0:
            self._pad_id = self.sp_model.pad_id()
            return self._pad_id

        pad_id = self.sp_model_vocab_size + self.used_special_tokens
        self.used_special_tokens += 1
        self._pad_id = pad_id
        return pad_id
