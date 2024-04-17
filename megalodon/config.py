from dataclasses import dataclass
from logging import getLogger
from typing import Optional
from pathlib import Path
from typing import List
import os

logger = getLogger()


@dataclass
class TokenizerConf:
    # config for processing different types of data
    path: Optional[str] = None
    additional_vocab_size: int = 0  # used for placeholder tokens
    data_tokenized: bool = False  # read already tokenized data

    @property
    def tokenizer_path(self) -> str:
        assert self.path is not None
        assert os.path.isfile(self.path), self.path
        return self.path


@dataclass
class OptimConf:
    lr: float = 1e-3
    warmup: int = 2000
    weight_decay: float = 0.1
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.98
    clip: float = 1.0
    scheduler: str = "linear"
    lr_init_ratio: float = 1e-4
    lr_end_ratio: float = 0.01
    cycles: float = 1.0


@dataclass
class ModelConf:
    num_layers: int = 8
    model_dim: int = 1024
    z_dim: int = 256
    value_dim: int = 2048
    num_heads: int = 1
    ffn_hidden_dim: int = 2048
    cema_ndim: int = 16
    chunk_size: int = 2048
    efficient_attn: Optional[str] = None
    init_mode: str = 'he'
    # input & output
    vocab_size: int = -1  # defined later by tokenizer
    output_size: int = -1
    # normalization
    norm_num_groups: int = 32
    norm_affine: bool = True
    norm_eps: float = 1e-5
    # rope base
    rope_base: Optional[float] = None
    # dropout rates
    dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    swiglu: bool = False
    rescale_nffn: bool = False
    scale_emb: bool = False
    share_emb: bool = False
    layerwise_ckpt: bool = False

    def __post_init__(self):
        assert self.z_dim % self.z_dim == 0
        assert self.value_dim % self.num_heads == 0

        assert 0 <= self.dropout < 1
        assert 0 <= self.attention_dropout < 1
        assert 0 <= self.hidden_dropout < 1

        assert self.efficient_attn in [None, "swift", "fused"]


@dataclass
class ValidConf:
    # tasks to evaluate
    ppl_files_str: str = ""  # comma separated list of files to eval PPL
    # prompts for generation tests
    prompt_path: str = ""  # prompt path

    batch_size: int = 1
    # decoding parameters
    use_sampling: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    gen_seq_len: int = 8192

    @property
    def ppl_files(self) -> List[str]:
        paths = [path for path in self.ppl_files_str.split(",") if len(path) > 0]
        return paths

    def __post_init__(self):
        # decoding params
        assert self.temperature >= 0
        assert self.top_k >= 0
        assert 0 <= self.top_p < 1

        # ppl files
        if len(self.ppl_files) > 0:
            assert len(self.ppl_files) == len(set(self.ppl_files))
            assert all(path.endswith(".jsonl") for path in self.ppl_files)
            assert all(Path(path).is_file() for path in self.ppl_files), self.ppl_files

        # generation task
        if self.prompt_path:
            assert self.prompt_path.endswith(".jsonl"), self.prompt_path
            assert os.path.isfile(self.prompt_path), self.prompt_path
