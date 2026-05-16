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
