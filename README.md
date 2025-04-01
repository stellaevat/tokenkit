<h1 align="center">tokenkit🔁</h1>
<h3 align="center">Tokenization Transfer for LLMs</h3>

![tokenkit](https://github.com/user-attachments/assets/a5d5e9a4-3df4-4552-aabc-400dcba1786e)

`tokenkit` is a toolkit implementing advanced methods to transfer *models* and *model knowledge* across tokenizers.

## News

- __2025-04-02__: The initial release of `tokenkit` with support for cross-tokenizer distillation via ALM and Zero-Shot Tokenizer Transfer via FVT!

## Contents
- [Why Transfer Across Tokenizers?](#why-transfer-across-tokenizers)
- [Installation](#installation)
- [Features](#features)
    - [Cross-Tokenizer Distillation](#cross-tokenizer-distillation)
    - [Zero-Shot Tokenizer Transfer](#zero-shot-tokenizer-transfer)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Why Transfer Across Tokenizers?

LLMs are bound to the tokenizer they were pretrained with. This limits their adaptability, reusability and modularity. Tokenizer transfer can lift this limitation. For example:
- If we want to reuse an LLM trained primarily on English in another language, we might want to update its tokenizer to one that is more suitable for the new language.
- If we want to combine (e.g., token-level ensemble) two LLMs, we need to transfer them to a common tokenizer.
- If we want to experiment with better tokenization schemes (e.g., byte-level tokenization), we might want to transfer an existing LLM to this tokenizer instead of training a new one expensively from scratch.
- If we want to transfer knowledge from a large teacher model to a smaller student model (which uses another tokenizer), we might want to use *cross-tokenizer distillation* to directly transfer the teacher's knowledge to the student without the need to first transfer the teacher to the student's tokenizer.

This library aims to let you accomplish all of this.

## Installation

`tokenkit` is primarily implemented in Jax, using PyTorch for data loading (so your PyTorch installation does not need to support an accelerator). Recommended installation:

```bash
# Clone the repository & install the library
git clone https://github.com/bminixhofer/tokenkit

# Create a new virtual environment
python -m venv tokenkit_env
. tokenkit_env/bin/activate

# Install torch & jax
# Jax installation instructions: https://docs.jax.dev/en/latest/installation.html#installation
# PyTorch installation instructions: https://pytorch.org/get-started/locally/
# For example:
pip install torch jax[tpu]==0.5.0

# Currently, tokenkit relies on forks of `transformers` and `lm_eval`
pip install git+https://github.com/bminixhofer/transformers
pip install git+https://github.com/bminixhofer/lm-evaluation-harness

# Install the library and the remaining dependencies
pip install -r requirements.txt
pip install -e .
pip install paxml==1.4.0 praxis==1.4.0 --no-deps
```

## Features

### Cross-Tokenizer Distillation

`tokenkit` supports [Approximate Likelihood Matching (ALM)](https://arxiv.org/abs/2503.20083) for cross-tokenizer distillation. We have found ALM to perform best, but we have implemented the following baselines:

- [Dual Space Knowledge Distillation (DSKD)](https://arxiv.org/abs/2406.17328)
- [Universal Logit Distillation (ULD)](https://arxiv.org/abs/2402.12030)
- [Minimum Edit Distance Logit Alignment (MinED)](https://arxiv.org/abs/2401.10491)

You can run cross-tokenizer distillation using the [`scripts/cross_tokenizer_distill.py`](scripts/cross_tokenizer_distill.py) script. See [`examples`](examples) for examples on transferring to different subword tokenizers and to byte-level tokenization.

### Zero-Shot Tokenizer Transfer

`tokenkit` supports Zero-Shot Tokenizer Transfer (ZeTT) via [Fast Vocabulary Transfer (FVT)](https://aclanthology.org/2022.emnlp-industry.41). Zero-Shot Tokenizer Transfer is usually used to obtain a good initialization for additional training, but can in some cases also be useful on its own. See our [ZeTT paper](https://arxiv.org/abs/2405.07883) for more details.

You can run Zero-Shot Tokenizer Transfer using the [`scripts/zett.py`](scripts/zett.py) script.

**🚧 We are working on implementing more ZeTT methods (including hypernetwork training introduced [here](https://arxiv.org/abs/2405.07883)). 🚧**

## Citation

To refer to this repository or to cite Approximate Likelihood Matching, please use this citation:

```
@article{alm,
  title={Cross-Tokenizer Distillation via Approximate Likelihood Matching},
  author={Minixhofer, Benjamin and Ponti, Edoardo Maria and Vuli{\'c}, Ivan},
  journal={arXiv preprint arXiv:2503.20083},
  year={2025}
}
```

Please use this citation for Zero-Shot Tokenizer Transfer:

```
@inproceedings{zett,
title={Zero-Shot Tokenizer Transfer},
author={Benjamin Minixhofer and Edoardo Ponti and Ivan Vuli{\'c}},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=RwBObRsIzC}
}
```

## Acknowledgments

Constituent projects (ALM, ZeTT) were supported by a Royal Society University Research Fellowship ‘Inclusive and Sustainable Language Technology for a Truly Multilingual World’ (no 221137; 2022-) awarded to Ivan Vulić, by the Google Cloud Research Credits program with the award GCP329647813, and by Cloud TPUs from Google’s TPU Research Cloud (TRC). The name `tokenkit` and the README layout were inspired by [mergekit](https://github.com/arcee-ai/mergekit).