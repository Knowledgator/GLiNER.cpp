# Basic exaqmple with small gliner model

To run the example, run CMake with -D BUILD_EXAMPLES=ON and download one of the ONNX gliner_small-v2.1 model to the project root, which can be downloaded here:

https://huggingface.co/onnx-community/gliner_small-v2.1/tree/main/onnx

To clone repository:

```bash
git lfs install
git clone https://huggingface.co/onnx-community/gliner_small-v2.1
```

To run example:

```bash
./build/examples/inference
```