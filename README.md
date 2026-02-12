# ANCHOR

ANCHOR is a **trajectory expansion framework** for scaling high-quality supervision for **end-to-end GUI agents** operating in real desktop environments. Starting from a small set of verified seed demonstrations, ANCHOR identifies meaningful **branch points**, proposes **state-grounded task variants** conditioned on the current GUI context, executes those variants to produce new trajectories, and applies **state-aware verification** and **denoising** to maintain coherent intent.


### What’s in this repo

- **`OSWorld/`**: OSWorld environment + our branch-expansion generator (`OSWorld/generate_branch_trajectories.py`). See `OSWorld/README.md` for setup, credentials, and usage.
- **`training_scripts/`**: standalone finetuning entrypoints for vision-language GUI agents trained on step-by-step trajectories.
  - `training_scripts/train_qwenvl.py`: finetune Qwen2/Qwen2.5/Qwen3-VL
  - `training_scripts/train_glm41v.py`: finetune GLM-4.1V
  - `training_scripts/train_utils.py`: dataset builders for branch-generated data (and an optional local “AgentNet human” format)

### Getting started

- **Generate expanded trajectories (OSWorld)**: follow `OSWorld/README.md`.
- **Finetune on the Anchor dataset**: follow `training_scripts/README.md`.

### Dataset (Hugging Face)

- **ANCHOR dataset**: [`Yale-nlp/ANCHOR`](https://huggingface.co/datasets/yale-nlp/Anchor)

### Benchmarks

- **OSWorld**: generation + environment code is included under `OSWorld/`.
- **WindowsAgentArena**: code will be released **soon**.

### Citation

If you use this repository or the ANCHOR dataset in your work, please cite:

- Paper: [*ANCHOR: Branch-Point Data Generation for GUI Agents* (arXiv:2602.07153)](https://arxiv.org/abs/2602.07153)

```bibtex
@misc{wei2026anchorbranchpointdatageneration,
  title         = {ANCHOR: Branch-Point Data Generation for GUI Agents},
  author        = {Wei, Jinbiao and Zhao, Yilun and Ni, Kangqi and Cohan, Arman},
  year          = {2026},
  eprint        = {2602.07153},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2602.07153},
}
```
