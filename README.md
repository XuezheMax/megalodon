<div align="center">
   <img src="./assets/logo.png" width="600"><br><br>
</div>

-----------------------------------------------

# Megalodon
Reference implementation of Megalodon 7B model.

>[Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length](https://arxiv.org/abs/2404.08801)

>Xuezhe Ma*, Xiaomeng Yang*, Wenhan Xiong, Beidi Chen, Lili Yu, Hao Zhang, Jonathan May, Luke Zettlemoyer, Omer Levy, Chunting Zhou*

Discord: [https://discord.gg/Unf8Fa7kWt](https://discord.gg/Unf8Fa7kWt)

## Updates
1. [April 15th 2024] Release Repo to public.

## Installation
First install PyTorch 2.0.1 with cuda 11.7
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Then, install the apex package.
```bash
# clone the repo
git clone https://github.com/NVIDIA/apex.git
cd apex

# checkout to the correct verison
git checkout 23.08

# complie & install
pip install packaging

# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Install the fairscale package.
```bash
# clone the repo 
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale

# checkout to the correct branch for bf16
git checkout ngoyal_bf16_changes

# install
pip install .
```

Finally, install megalodon
```bash
https://github.com/XuezheMax/megalodon.git
cd megalodon
pip install -r requirements.txt
pip install -e .
```

## Evaluating Pretrained LLMs
To launch an evaluation job, we recommend to use `torchrun` or `slurm`. We provide an example script of `torchrun`
```bash
export NGPU=<NUM_GPUS>; torchrun --nproc_per_node=$NGPU eval.py \
        --model_parallel_size 1 \
        --checkpoint_dir <Checkpoint Dir> \
        --tokenizer_path <Tokenizer Path> \ # remove it if using origianl model tokenizer
        --dump_dir <Dump Dir> \ # diretory where results are dumped
        --dtype "bf16" \ # default is bf16
        --master_logging_only "true" \ # only print logs from master
        --ppl_files_str <PPL Files> \ # comma separated list of files to eval PPL
        --prompt_path <Prompt File Path> \
        --batch_size <Batch Size>
```
All the data should be prepared in `jsonl` format where the content texts are in the `text` field of each json item.

## Launching a job for LLM pretraining
We provide a pseudo code for LLM pretraining
```python
from logging import getLogger
from megalodon.logger import initialize_logger
from megalodon.distributed import (
    init_signal_handler,
    init_torch_distributed,
    initialize_model_parallel,
)
from megalodon.model.mega import build_model
from megalodon.optim import build_optimizer
from megalodon.modules.losses import cross_entropy
from megalodon.utils import (
    setup_env,
    log_host,
    clip_grad_norm_,
    set_random_seed,
)

logger = getLogger()

initialize_logger()
setup_env()
log_host()

cfg = TrainerConf()  # training config

init_signal_handler()

# initialize distributed mode / model parallel
logger.info("Starting init of torch.distributed...")
slurm_cfg = init_torch_distributed()
logger.info("Done init of torch.distributed.")

logger.info("Starting init of model parallel...")
initialize_model_parallel(cfg.model_parallel_size, cfg.chunk_parallel_size)
logger.info("Done init of model parallel.")

logger.info(
    f"Global rank: {slurm_cfg.global_rank} -- "
    f"model parallel rank: {cfg.model_parallel_rank}/{cfg.model_parallel_size} -- "
    f"chunk parallel rank: {cfg.chunk_parallel_rank}/{cfg.chunk_parallel_size} -- "
    f"data  parallel rank: {cfg.data_parallel_rank}/{cfg.data_parallel_size}"
)

dataloader = DataLoader()

logger.info("Start building of model...")
model = build_model(cfg.model, dtype=cfg.dtype,
                    fp32_reduce_scatter=cfg.fp32_reduce_scatter,
                    reshard_after_forward=cfg.reshard_after_forward)
logger.info(model)
model.train()

# build optimizer / scheduler
optimizer, scheduler = build_optimizer(model, cfg.optim, cfg.steps, cfg.dtype)

for batch in dataloader:
    x, y = batch
    pred, _ = model(x)  # forward pass
    tok_loss = cross_entropy(pred, y)
    loss = tok_loss.mean()
    
    loss.backward()  # backward pass
    model.grad_all_reduce()  # sync grad across each chunk parallel group
    clip_grad_norm_(fsdp_module=model, max_norm=cfg.optim.clip)  # grad clip

    optimizer.step()  # optimizer step
    scheduler.step()  # scheduler step
    optimizer.zero_grad()

```

## References
```
@misc{ma2024megalodon,
      title={Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length}, 
      author={Xuezhe Ma and Xiaomeng Yang and Wenhan Xiong and Beidi Chen and Lili Yu and Hao Zhang and Jonathan May and Luke Zettlemoyer and Omer Levy and Chunting Zhou},
      year={2024},
      eprint={2404.08801},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@inproceedings{
  ma2023mega,
  title={Mega: Moving Average Equipped Gated Attention},
  author={Xuezhe Ma and Chunting Zhou and Xiang Kong and Junxian He and Liangke Gui and Graham Neubig and Jonathan May and Luke Zettlemoyer},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
}
```