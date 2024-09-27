# Basic Example with Small Gliner Model

Before running the example, you need to download the ONNX runtime for your system.

For .tar.gz files, you can use the following command:

```bash
tar -xvzf onnxruntime-linux-x64-1.19.2.tgz 
```

## Run the Example

You need to provide the ONNXRUNTIME_ROOTDIR option, which should be set to the absolute path of the chosen ONNX runtime.
In this example, /home/usr/onnxruntime-linux-x64-1.19.2 is used.

Build the example and run:

```bash
./download-gliner_small-v2.1.sh
cmake -D ONNXRUNTIME_ROOTDIR="/home/usr/onnxruntime-linux-x64-1.19.2" -D GPU_CHECK=ON -S . -B build
cmake --build build --target inference -j
./build/inference
```

This will download the model files if they are not already downloaded, build the project, and run the inference.

## Run the Example on GPU

By default, the CPU is used. To use the GPU, you need to utilize the ONNX runtime with GPU support and set up cuDNN.
Follow the instructions to install cuDNN here:

https://developer.nvidia.com/cudnn-downloads

Build the example with the ONNX runtime with GPU support and run:

```bash
./download-gliner_small-v2.1.sh
cmake -D ONNXRUNTIME_ROOTDIR="/home/usr/onnxruntime-linux-x64-gpu-1.19.2" -S . -B build
cmake --build build --target inference_gpu -j
./build/inference_gpu
```

To use GPU:

- You need to specify it in this contructor using 'device_id' (used in example):

```c++
int device_id = 0 // (CUDA:0)
gliner::Model model("./gliner_small-v2.1/onnx/model.onnx", config, processor, decoder, 0); // device_id = 0 (CUDA:0)
```

OR

- Use custom environment(Ort::Env) and session options(Ort::SessionOptions) of the ONNX runtime: 

```c++
Ort::Env env = ...;
Ort::SessionOptions session_options = ...;
gliner::Model model("./gliner_small-v2.1/onnx/model.onnx", config, processor, decoder, env, session_options);
```

## Model

This example uses the gliner_small-v2.1 ONNX model, which can be found here:

https://huggingface.co/onnx-community/gliner_small-v2.1/tree/main