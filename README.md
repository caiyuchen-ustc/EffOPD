# EffOPD

This repository contains the implementation of **EffOPD**.

The analysis in the earlier part of our paper uses the code in the `analysis` folder.

## Codebase

EffOPD is implemented based on verl and GOPD. We mainly modify the following files:

- `ppo_trainer.yaml`
- `fsdp_workers.py`
- `ray_trainer.py`

## Training EffOPD

The training dataset can be download from : https://huggingface.co/datasets/Keven16/G-OPD-Training-Data

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
