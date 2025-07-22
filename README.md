# Cheetah
Fast distributed gen AI model inference

## Background
After issues with the exo fork xotorch dealing with inference speed, we decided to focus on a more low level approach using the PyTorch C++ library. This code is mostly a conversion of the generalized Multi-Head Attention model into C++ PyTorch Frontend to be integrated into [xotorch](https://github.com/shamantechnology/xotorch) PytorchShardedInferenceEngine. This is an early stage project but PRs are welcomed at any point.

## Features
Right now we are focusing on CPU based inference along with using large lanaguage models.
- [ ] Multi-Head Attention based Model Inference
- [ ] Mixture of Experts MHA based Model Inference
- [ ] Integration into [xotorch](https://github.com/shamantechnology/xotorch)

## Building for CPU
For Linux and MacOS

*Still developing an automated install.* 
- Download a copy of the [pytorch](https://pytorch.org/) C++ library for CPU and unzip it in the "libs" folder.

```console
$ mkdir build && cd build
$ cmake .. -DCMAKE_PREFIX_PATH=../cheetah/libs/libtorch
$ cmake --build .
$ ./main
```

To manually set the sequence length, use the environment variable
```console
CHEETAH_MAX_SEQ_LEN
```

## Building for NVIDIA
Work in progress

## Building for AMD
Work in progress
