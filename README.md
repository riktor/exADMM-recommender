# exADMM
This repository contains the code used for the experiments in ["Scalable and Provably Fair Exposure Control for Large-Scale Recommender Systems"](#) by [Riku Togashi](https://riktor.github.io/), [Kenshi Abe](https://bakanaouji.github.io/), and [Yuta Saito](https://usait0.com/en/).


## Citation
```
@inproceedings{togashi2024scalable,
author = {Togashi, Riku and Abe, Kenshi and Saito, Yuta},
title = {Scalable and Provably Fair Exposure Control for Large-Scale Recommender Systems},
year = {2024},
booktitle = {Proceedings of the ACM Web Conference 2024},
}
```

## Setup

### Build all-in-one docker image.
Build docker image and generate datasets inside the image.
```sh
$ sh build_image.sh
```

### Build executable without docker.
The code can be used without docker.
See Dockerfile for an example installation, which contains the minimum setup instructions for a debian-based image (slim).

Build executable locally. Need to install [bazel](https://github.com/bazelbuild/bazel) and gcc/g++-9 (or later versions).
```sh
$ bazel build run_model
```

Create convenient symlinks (optional).
```sh
$ ln -s $(bazel info bazel-bin) bazel-bin
```

Install dependency.
```sh
$ pip install -r scripts/requirements.txt
```

Download and generate datasets.
```sh
$ python scripts/generate_data.py
```

### Testing
Run tests manually for iALS and exADMM.
```sh
$ bazel test ials_test exadmm_test --test_output=all
```
These tests run automatically through github workflow.
See `.github/workflow/bazel.yaml` for details.


## Run Models inside Docker Container
### Epinions
exADMM
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name exadmm --train_data epinions/train.csv --test_train_data epinions/validation_tr.csv --test_test_data epinions/validation_te.csv --dim 32 --alpha 0.06 --l2_reg 2e-4 --epoch 50 --pred_iterations 50 --admm_penalty 6e-2 --exposure_reg 1e-3 --learning_rate 0.01 --print_train_stats 1
```

iALS
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name ials --train_data epinions/train.csv --test_train_data epinions/validation_tr.csv --test_test_data epinions/validation_te.csv --dim 32 --alpha 0.06 --l2_reg 7e-4 --epoch 50 --print_train_stats 1
```

### ML-20M
exADMM
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name exadmm --train_data ml-20m/train.csv --test_train_data ml-20m/validation_tr.csv --test_test_data ml-20m/validation_te.csv --dim 256 --alpha 0.1 --l2_reg 0.002 --epoch 50 --pred_iterations 50 --admm_penalty 5e-7 --exposure_reg 5e-8 --learning_rate 0.01 --print_train_stats 1
```

iALS
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name ials --train_data ml-20m/train.csv --test_train_data ml-20m/validation_tr.csv --test_test_data ml-20m/validation_te.csv --dim 256 --alpha 0.1 --l2_reg 0.003 --epoch 50 --print_train_stats 1
```

### MSD
exADMM
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name exadmm --train_data msd/train.csv --test_train_data msd/validation_tr.csv --test_test_data msd/validation_te.csv --dim 512 --alpha 0.02 --l2_reg 0.002 --epoch 50 --pred_iterations 50 --admm_penalty 3e-6 --exposure_reg 2e-7 --learning_rate 0.01 --print_train_stats 1
```

iALS
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name ials --train_data msd/train.csv --test_train_data msd/validation_tr.csv --test_test_data msd/validation_te.csv --dim 512 --alpha 0.03 --l2_reg 0.0005 --epoch 50 --print_train_stats 1
```

## Directory Structure
Following is the directory structure of this repository,
which may be helpful to read through the code.

```
.
├── build_image.sh          (for building reproducible docker image)
├── Dockerfile
├── .bazelrc                (bazel configuration)
│── WORKSPACE               (for dependency)
├── BUILD                   (for C++ compilation options and linking)
├── bazel
│   └── frecsys.bzl
├── 3rdparty
│   └── eigen.BUILD         (for using the latest Eigen)
├── include                 (**header-only implementation**)
│   └── frecsys
│       ├── types.hpp       (type definition)
│       ├── dataset.hpp     (struct definition of dataset)
│       ├── evaluation.hpp  (struct definition of evaluation results)
│       ├── recommender.hpp (definition of the abstract base class)
│       ├── ials.hpp        (implementation of iALS)
│       ├── exadmm.hpp      (implementation of exADMM)
│       └── fairrec.hpp     (implementation of FairRec)
├── README.md
├── scripts                 (**miscellaneous scripts**)
│   ├── requirements.txt    (python dependency)
│   └── generate_data.py    (script for preparing datasets)
├── tests
│   ├── ials_test.cpp       (integration test of iALS)
│   ├── exadmm_test.cpp      (integration test of exADMM)
│   ├── ml-1m               (sample dataset for integration testing)
│   │   ├── train.csv
│   │   ├── validation_te.csv
│   └── └── validation_tr.csv
├── tools                   (**executable implementation**)
│   ├── CLI11               (dependency for CLI)
│   │   └── CLI11.hpp
└── └── run_model.cpp       (implementation of CLI)
```
