I'm using this to train some simple base -> SFT models for my  work


```sh
# get a H100 V100 from your favorite cloud provider

# create virtualenv
uv sync --verbose
# took me ~30mins due to building flash-attn


# MAX_JOBS=10 pip install flash-attn --no-build-isolation
#uv add --no-build-isolation-package flash-attn
. ./.venv/bin/activate

# note  --num_processes=1 related to the GPU's you have
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes=1 scripts/run_sft.py recipes/fromSimPO/Qwen3-0.6B_fourchan.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/fromSimPO/Qwen3-0.6B_fourchan.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/fromSimPO/SmolLM2-135M.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/fromSimPO/SmolLM2-360M.yaml
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/fromSimPO/Qwen3-0.6B_fourchan.yaml


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/fromSimPO/Qwen3-0.6B-sft.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/fromSimPO/llama-3-2-3b-base-sft.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 recipes/fromSimPO/Qwen3-4B-fourchan.yaml
recipes/fromSimPO/SmolLM2-1.7B-sft.yaml

```


Old readme
----
<p align="center">
  <img src="https://raw.githubusercontent.com/huggingface/alignment-handbook/main/assets/handbook.png">
</p>

<p align="center">
    🤗 <a href="https://huggingface.co/collections/alignment-handbook/handbook-v01-models-and-datasets-654e424d22e6880da5ebc015" target="_blank">Models & Datasets</a> | 📃 <a href="https://arxiv.org/abs/2310.16944" target="_blank">Technical Report</a>
</p>

# The Alignment Handbook

Robust recipes to continue pretraining and to align language models with human and AI preferences.

## What is this?

Just one year ago, chatbots were out of fashion and most people hadn't heard about techniques like Reinforcement Learning from Human Feedback (RLHF) to align language models with human preferences. Then, OpenAI broke the internet with ChatGPT and Meta followed suit by releasing the Llama series of language models which enabled the ML community to build their very own capable chatbots. This has led to a rich ecosystem of datasets and models that have mostly focused on teaching language models to follow instructions through supervised fine-tuning (SFT).

However, we know from the [InstructGPT](https://huggingface.co/papers/2203.02155) and [Llama2](https://huggingface.co/papers/2307.09288) papers that significant gains in helpfulness and safety can be had by augmenting SFT with human (or AI) preferences. At the same time, aligning language models to a set of preferences is a fairly novel idea and there are few public resources available on how to train these models, what data to collect, and what metrics to measure for best downstream performance.

The Alignment Handbook aims to fill that gap by providing the community with a series of robust training recipes that span the whole pipeline.

## News 🗞️
* **November 21, 2024**: We release the [recipe](recipes/smollm2/README.md) for fine-tuning SmolLM2-Instruct.
* **August 18, 2024**: We release SmolLM-Instruct v0.2, along with the [recipe](recipes/smollm/README.md)  to fine-tuning small LLMs 💻
* **April 12, 2024**: We release Zephyr 141B (A35B), in collaboration with Argilla and Kaist AI, along with the recipe to fine-tune Mixtral 8x22B with ORPO 🪁
* **March 12, 2024:** We release StarChat2 15B, along with the recipe to train capable coding assistants 🌟
* **March 1, 2024:** We release Zephyr 7B Gemma, which is a new recipe to align Gemma 7B with RLAIF 🔥
* **February 1, 2024:** We release a recipe to align open LLMs with Constitutional AI 📜! See the [recipe](https://github.com/huggingface/alignment-handbook/tree/main/recipes/constitutional-ai) and the [blog post](https://huggingface.co/blog/constitutional_ai) for details. 
* **January 18, 2024:** We release a suite of evaluations of DPO vs KTO vs IPO, see the [recipe](recipes/pref_align_scan/README.md) and the [blog post](https://huggingface.co/blog/pref-tuning) for details.
* **November 10, 2023:** We release all the training code to replicate Zephyr-7b-β 🪁! We also release [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots), a brand new dataset of 10,000 instructions and demonstrations written entirely by skilled human annotators.

## Links 🔗

* [Zephyr 7B models, datasets, and demos](https://huggingface.co/collections/HuggingFaceH4/zephyr-7b-6538c6d6d5ddd1cbb1744a66)

## How to navigate this project 🧭

This project is simple by design and mostly consists of:

* [`scripts`](./scripts/) to train and evaluate models. Four steps are included: continued pretraining, supervised-finetuning (SFT) for chat, preference alignment with DPO, and supervised-finetuning with preference alignment with ORPO. Each script supports distributed training of the full model weights with DeepSpeed ZeRO-3, or LoRA/QLoRA for parameter-efficient fine-tuning.
* [`recipes`](./recipes/) to reproduce models like Zephyr 7B. Each recipe takes the form of a YAML file which contains all the parameters associated with a single training run. A `gpt2-nl` recipe is also given to illustrate how this handbook can be used for language or domain adaptation, e.g. by continuing to pretrain on a different language, and then SFT and DPO tuning the result. 

We are also working on a series of guides to explain how methods like direct preference optimization (DPO) work, along with lessons learned from gathering human preferences in practice. To get started, we recommend the following:

1. Follow the [installation instructions](#installation-instructions) to set up your environment etc.
2. Replicate Zephyr-7b-β by following the [recipe instructions](./recipes/zephyr-7b-beta/README.md).

If you would like to train chat models on your own datasets, we recommend following the dataset formatting instructions [here](./scripts/README.md#fine-tuning-on-your-datasets).


## Contents

The initial release of the handbook will focus on the following techniques:

* **Continued pretraining:** adapt language models to a new language or domain, or simply improve it by continued pretraining (causal language modeling) on a new dataset.
* **Supervised fine-tuning:** teach language models to follow instructions and tips on how to collect and curate your training dataset.
* **Reward modeling:** teach language models to distinguish model responses according to human or AI preferences.
* **Rejection sampling:** a simple, but powerful technique to boost the performance of your SFT model.
* **Direct preference optimisation (DPO):** a powerful and promising alternative to PPO.
* **Odds Ratio Preference Optimisation (ORPO)**: a technique to fine-tune language models with human preferences, combining SFT and DPO in a single stage.

## Installation instructions

To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n handbook python=3.10 && conda activate handbook
```

Next, install PyTorch `v2.1.2` - the precise version is important for reproducibility! Since this is hardware-dependent, we
direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/locally/).

You can then install the remaining package dependencies as follows:

```shell
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
python -m pip install flash-attn --no-build-isolation
```

> **Note**
> If your machine has less than 96GB of RAM and many CPU cores, reduce the `MAX_JOBS` arguments, e.g. `MAX_JOBS=4 pip install flash-attn --no-build-isolation`

Next, log into your Hugging Face account as follows:

```shell
huggingface-cli login
```

Finally, install Git LFS so that you can push models to the Hugging Face Hub:

```shell
sudo apt-get install git-lfs
```

You can now check out the `scripts` and `recipes` directories for instructions on how to train some models 🪁!

## Project structure

```
├── LICENSE
├── Makefile                    <- Makefile with commands like `make style`
├── README.md                   <- The top-level README for developers using this project
├── chapters                    <- Educational content to render on hf.co/learn
├── recipes                     <- Recipe configs, accelerate configs, slurm scripts
├── scripts                     <- Scripts to train and evaluate chat models
├── setup.cfg                   <- Installation config (mostly used for configuring code quality & tests)
├── setup.py                    <- Makes project pip installable (pip install -e .) so `alignment` can be imported
├── src                         <- Source code for use in this project
└── tests                       <- Unit tests
```

## Citation

If you find the content of this repo useful in your work, please cite it as follows via `\usepackage{biblatex}`:

```bibtex
@software{Tunstall_The_Alignment_Handbook,
  author = {Tunstall, Lewis and Beeching, Edward and Lambert, Nathan and Rajani, Nazneen and Huang, Shengyi and Rasul, Kashif and Bartolome, Alvaro and M. Rush, Alexander and Wolf, Thomas},
  license = {Apache-2.0},
  title = {{The Alignment Handbook}},
  url = {https://github.com/huggingface/alignment-handbook},
  version = {0.3.0.dev0}
}
```
