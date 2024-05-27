We use <a href="https://github.com/hiyouga/LLaMA-Factory/tree/main">[LLaMA-Factory] tool for model finetuning.

We use NUCLE and W\&I (BEA) datasets for model finetuning.

The code was modified to use custom system prompt.


## Training scripts located in the directory:

* Full supervised tuning

`LLaMA-Factory/examples/full_multi_gpu/run.sh`

* PERF (LoRA) supervised tuning

`LLaMA-Factory/examples/lora_multi_gpu/run.sh` - supervised tuning of different LLMs

`LLaMA-Factory/examples/lora_multi_gpu/run-dpo.sh`  and `LLaMA-Factory/examples/lora_multi_gpu/run-dpo.sh`  - DPO model tuning with preference datasets

## Data

Training GEC dataset `LLaMA-Factory/data/gec_dataset/` and `LLaMA-Factory/data/gec_bea_train.json`

NUCLE testset `LLaMA-Factory/data/gec_nucle_test.json`

BEA-2019-dev testset `LLaMA-Factory/data/gec_bea_dev.json`