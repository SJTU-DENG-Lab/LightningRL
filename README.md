<div align="center">

<p align="center">
  <img src="assets/logo_text.png" alt="LightningRL" width="420">
</p>

<h3>Breaking the Accuracy–Parallelism Trade-off of Block-wise dLLMs via Reinforcement Learning</h3>

<p>
<b>Yanzhe Hu</b><sup>1,2</sup>, <b>Yijie Jin</b><sup>1</sup>, <b>Pengfei Liu</b><sup>1</sup>, <b>Kai Yu</b><sup>1</sup>, <b>Zhijie Deng</b><sup>1,†</sup>
</p>

<p>
<sup>1</sup>Shanghai Jiao Tong University    <sup>2</sup>Huazhong University of Science and Technology
</p>

<p>
<sup>†</sup>Corresponding author
</p>

</div>

<p align="center">  
  <a href="#">
    <img src="https://img.shields.io/badge/Website-LightningRL-purple.svg" alt="ICML 2026"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg" alt="Paper on arXiv"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/GitHub-Code-black.svg?logo=github" alt="GitHub Code"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow.svg" alt="Hugging Face Model"/>
  </a>
</p>

<p align="center">
  <img src="assets/figure1.png" alt="Overview" width="750">
</p>

---

## TL;DR

We propose **LightningRL**, a reinforcement learning framework that breaks the accuracy–parallelism trade-off of block-wise diffusion Large Language Models (dLLMs). LightningRL optimizes both speed and generation quality simultaneously through three key modifications to GRPO: **per-reward decoupled normalization**, **token-level NLL regularization**, and **TPF-aware filtering**. Applied to SDAR-8B, LightningRL achieves an average TPF of **7.32** and AUP of **497.9**, significantly outperforming EAGLE-3, Fast-dLLM-v2, and other leading baselines across math and code benchmarks.

## Highlights

- **Breaking the Trade-off**: LightningRL achieves **7.32** average TPF and **497.9** AUP, simultaneously improving both speed and accuracy of block-wise dLLMs
- **Three Key Innovations**: Per-reward decoupled normalization, token-level NLL loss, and TPF-aware filtering work synergistically to stabilize multi-objective RL training
- **Strong Generalization**: Consistent improvements across math (GSM8K, MATH500) and code (MBPP, HumanEval) benchmarks
- **Practical Speed**: **336.03** TPS on H100 GPUs, 3.2x faster than the SDAR baseline

## Method

We advocate a post-training approach for pre-trained block-wise dLLMs that directly optimizes the speed–quality frontier. Our core insight is that we do not require the model to decode aggressively along all sampling trajectories, but rather to find several highly parallelizable ones that can yield correct results. We formulate this as a reinforcement learning problem using the GRPO framework with three key modifications:

<p align="center">
  <img src="assets/figure2.png" alt="LightningRL Overview" width="750">
</p>

- **Per-Reward Decoupled Normalization**: Independently normalizes each reward component within the group to prevent signal collapse when combining accuracy and speed rewards of different scales
- **Token-Level NLL Regularization**: Applies dense token-factorized supervision on correct trajectories to anchor the policy toward correctness and prevent drift into fast-but-incorrect modes
- **TPF-Aware Filtering**: Dynamically selects prompts whose sampled trajectories exhibit diverse levels of parallelism, maintaining meaningful learning signals and improving sample efficiency

## Performance

### LightningRL vs RL Methods

Compared with existing RL approaches for dLLMs (TraceRL, GRPO), LightningRL delivers substantial improvements across both math and code benchmarks, achieving the best Acc, TPF, and AUP simultaneously.

<p align="center">
  <img src="assets/table1.png" alt="Table 1: LightningRL vs RL Methods" width="750">
</p>

### Full Comparison with Baselines

LightningRL consistently advances the Pareto frontier against all categories of baselines — vanilla dLLMs (Dream, LLaDA), AR models (Qwen, EAGLE-3), and block-wise dLLMs (Fast-dLLM-v2, SDAR).

<p align="center">
  <img src="assets/table2.png" alt="Table 2: Full Comparison" width="750">
</p>

### Wall-Clock Speed (H100 GPUs)

LightningRL achieves **336.03** TPS on a single H100 GPU, **3.2x** faster than the SDAR baseline and significantly outperforming all other methods while maintaining the highest accuracy (**90.3%**).

<p align="center">
  <img src="assets/table7.png" alt="Table 7: Wall-Clock Speed" width="750">
</p>

## Quick Start

### Installation

```bash
git clone https://github.com/SJTU-DENG-Lab/LightningRL.git

conda create --name lightningrl python=3.10
conda activate lightningrl

pip install torch==2.6.0
pip install -r requirements.txt
```

### Training

LightningRL post-training on SDAR-8B:

```bash
python train.py config=configs/rl.yaml
```

### Evaluation

```bash
python eval.py config=configs/eval.yaml
```

## ⚙️ Data

You can navigate to `./data` to download datasets for evaluation and training:

```bash
cd data
python download_data.py --dataset MATH500
python download_data.py --dataset MATH_train
cd ..
```

After downloading the data, select (or create) a config file in `./configs` to specify the dataset paths and training settings.

## Acknowledgement

We would like to express our gratitude to the following works for providing important foundations and inspiration:

[SDAR](https://github.com/JetAstra/SDAR), [dLLM-RL](https://github.com/Gen-Verse/dLLM-RL), [Block Diffusion](https://arxiv.org/abs/2503.09573), [DiRL](https://github.com/OpenMOSS/DiRL), [lmdeploy](https://github.com/InternLM/lmdeploy),.

## Contact

For issues or inquiries:

- **Yanzhe Hu**, Huazhong University of Science and Technology ([yanzhehu@hust.edu.cn](mailto:yanzhehu@hust.edu.cn))

## Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{hu2026lightningrl,
  title={LightningRL: Breaking the Accuracy--Parallelism Trade-off of Block-wise dLLMs via Reinforcement Learning},
  author={Hu, Yanzhe and Jin, Yijie and Liu, Pengfei and Yu, Kai and Deng, Zhijie},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2026}
}
```
