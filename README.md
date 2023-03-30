# 土石坝渗流监测数据的深度学习方法研究

This repository contains the code for the reproducibility of the experiments presented in the paper "Study on Deep Learning-based Imputation Method for Monitoring Data of Embankment Dam". We propose a graph neural network that exploits spatio-temporal dependencies to impute missing values leveraging only valid observations.

**Authors**: Pan Liao, Xiaoqing Li

## Directory structure

The directory is structured as follows:

```
.
├── config/
│   ├── imputation/
│   │   ├── brits.yaml
│   │   ├── bigan.yaml
│   │   ├── grin.yaml
│   │   ├── gpvae.yaml
│   └── inference.yaml
├── spin/
│   ├── baselines/
│   ├── imputers/
│   ├── layers/
│   ├── models/
│   └── ...
├── conda_env.yaml
└── run_imputation.py
└── run_inference.py
└── tsl_config.yaml

```

## Installation

We provide a requirements list with all the project dependencies in `requirements.txt`. To install the environment use:

## Configuration files

The `config/` directory stores all the configuration files used to run the experiment. `config/imputation/` stores model configurations used for experiments on imputation.

## Experiments

The scripts used for the experiment in the paper are in the `run_imputation.py` and `run_inference.py`.

* `run_imputation.py` is used to compute the metrics for the deep imputation methods. An example of usage for BRITS model is

	```bash
	python run_imputation.py --config imputation/brits.yaml --model-name brits --dataset-name pwp
	```

* `run_inference.py` is used for the experiments on sparse datasets using pre-trained models. An example of usage is

	```bash
	python run_inference.py --config inference.yaml --model-name brits --dataset-name pwp --exp-name {exp_name}
	```

 