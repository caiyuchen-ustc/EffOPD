# EffOPD

This repository contains the implementation of **EffOPD**.

The analysis in the earlier part of our paper uses the code in the `Analysis` folder.

## Codebase

EffOPD is implemented based on [verl](https://github.com/volcengine/verl) and [G-OPD](https://github.com/RUCBM/G-OPD). We mainly modify the following files:

- `ppo_trainer.yaml`
- `fsdp_workers.py`
- `ray_trainer.py`

## Installation

Our code is mainly based on [verl](https://github.com/volcengine/verl) (v0.6.1) and [G-OPD](https://github.com/RUCBM/G-OPD). To prepare the environment, please follow these steps from the repository root:

```bash
conda create -n effopd python=3.10 -y
conda activate effopd
cd EffOPD/verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
python -m pip install math-verify
python -m pip install --no-deps -e .
```

## Training EffOPD

The training dataset can be downloaded from [G-OPD-Training-Data](https://huggingface.co/datasets/Keven16/G-OPD-Training-Data).

To launch EffOPD training, please start from the original bash script used for training OPD, and add the following arguments:

```bash
trainer.enable_iterative_test=True \
trainer.max_test_iterations=5 \
data.iterative_test_files=xxx.parquet
```

Here:

- `trainer.enable_iterative_test=True` enables the EffOPD extrapolation search.
- `trainer.max_test_iterations=5` sets the maximum number of extrapolated candidate parameters to evaluate at each exponential checkpoint. In our experiments, this value is set to `5`.
- `data.iterative_test_files=xxx.parquet` specifies the data file used to construct the lightweight validation set for immediate validation. Please replace `xxx.parquet` with the actual path to the validation parquet file.

### Prepare quick-validation data

Run the following commands from the repository root before training. Each command samples 50 training examples for evaluating EffOPD extrapolation candidates.

**Math:**

```bash
python EffOPD/verl/scripts/subdata.py \
  --input data/DeepMath-103K/train_filtered_level6.parquet \
  --output data/quick_validation/math_validation.parquet \
  --num-samples 50 --sampling random
```

**Code:**

```bash
python EffOPD/verl/scripts/subdata.py \
  --input data/Eurus/code_train.parquet \
  --output data/quick_validation/code_validation.parquet \
  --num-samples 50 --sampling random
```

Existing output files are reused unchanged. Training reads these files and does not generate them automatically. Regular evaluation uses AIME 2024/2025 for math and the Eurus code validation set for code.

### Training scripts

Run from the repository root. Models and datasets default to `models/` and `data/`; use `STUDENT_MODEL`, `TEACHER_MODEL`, `TRAIN_FILE`, `VAL_FILES`, and `QUICK_VAL_FILE` to override paths. Both scripts use the quick-validation files prepared above.

**Qwen3-4B math:** distill Qwen3-4B-Non-Thinking-RL-Math into Qwen3-4B on DeepMath with EffOPD.

```bash
bash EffOPD/verl/examples/effopd/train_qwen3_4b_math.sh
```

**Qwen-1.5B code:** distill Nemotron-Research-Reasoning-Qwen-1.5B into DeepSeek-R1-Distill-Qwen-1.5B on Eurus code data with EffOPD.

```bash
bash EffOPD/verl/examples/effopd/train_qwen_1_5b_code.sh
```

If you find this project interesting, feel free to ⭐ star the repository or open an issue for discussion!

If you use this code in your research, please cite:

```bibtex
@article{cai2026learning,
  title={Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation},
  author={Cai, Yuchen and Cao, Ding and Lin, Liang and Luo, Chunxi and Xu, Xin and Yang, Kai and Liu, Weijie and Yang, Saiyong and Zhao, Tianxiang and Sun, Guangzhong and others},
  journal={arXiv preprint arXiv:2605.11739},
  year={2026}
}

@article{cai2025predictability,
  title={On Predictability of Reinforcement Learning Dynamics for Large Language Models},
  author={Cai, Yuchen and Cao, Ding and Xu, Xin and Yao, Zijun and Huang, Yuqing and Tan, Zhenyu and Zhang, Benyi and Liu, Guiquan and Fang, Junfeng},
  journal={arXiv preprint arXiv:2510.00553},
  year={2025}
}
```
