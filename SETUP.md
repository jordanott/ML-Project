# Setup

## Requirements
Python 2.7  
Pytorch 0.4  
Cuda 9  
Centos 7  
NVIDIA P100  

## CTC Loss ##

You'll need to install a package that allows us to compute the CTC loss in Pytorch. Follow the instructions [here](https://github.com/SeanNaren/warp-ctc).  

I got a gcc error when running:
```
python setup.py install
```
To fix it I made the following modifications:
```
# "pytorch_binding/src/binding.cpp"
# line 92
int probs_size = THCudaTensor_size(state, probs, 2);
# line 105
void* gpu_workspace;
THCudaMalloc(state, &gpu_workspace, gpu_size_bytes);
```
