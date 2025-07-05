# Cheetah
Fast distributed gen AI model inference

## Background
After issues with the exo fork xotorch dealing with inference speed, we decided to focus on a more low level approach using the PyTorch C++ library. This code is mostly a conversion of the PytorchInferenceEngine into C++ along with the distributed features of xotorch. This is an early stage project but PRs are welcomed at any point.

## Features
Right now we are focusing on CPU based inference along with using large lanaguage models.
- [ ] Large Language Model Inference Engine
- [ ] Network Discovery
- [ ] Distributed Tensor Compute
- [ ] TUI

## Building for CPU
For Linux and MacOS

Creating a automated install. 
- Download a copy of the [pytorch](https://pytorch.org/) C++ library for CPU and unzip it in the "libs" folder.

```console
$ mkdir build && cd build
$ cmake .. -DCMAKE_PREFIX_PATH=../cheetah/libs/libtorch
$ cmake --build .
$ ./main
```

## Building for NVIDIA
Work in progress

## Building for AMD
Work in progress