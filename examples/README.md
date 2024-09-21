# Basic Example with Small Gliner Model

Before running the example, you need to download the ONNX runtime for your system.

For .tar.gz files, you can use the following command:

```bash
tar -xvzf onnxruntime-linux-x64-1.19.2.tgz 
```

## Run the Example

You need to provide the ONNXRUNTIME_ROOTDIR option, which should be set to the absolute path of the chosen ONNX runtime.
In this example, /home/usr/onnxruntime-linux-x64-1.19.2 is used.

```bash
./download-gliner_small-v2.1.sh
cmake -D ONNXRUNTIME_ROOTDIR="/home/usr/onnxruntime-linux-x64-1.19.2" -S . -B build
cmake --build build --target all -j
./build/inference
```

This will download the model files if they are not already downloaded, build the project, and run the inference.

## Model

This example uses the gliner_small-v2.1 ONNX model, which can be found here:

https://huggingface.co/onnx-community/gliner_small-v2.1/tree/main