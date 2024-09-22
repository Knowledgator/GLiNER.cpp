
mkdir -p gliner_small-v2.1
cd gliner_small-v2.1
if [ ! -f "tokenizer.json" ]; then
    wget https://huggingface.co/onnx-community/gliner_small-v2.1/resolve/main/tokenizer.json
fi
mkdir -p onnx
cd onnx
if [ ! -f "model.onnx" ]; then
    wget https://huggingface.co/onnx-community/gliner_small-v2.1/resolve/main/onnx/model.onnx
fi