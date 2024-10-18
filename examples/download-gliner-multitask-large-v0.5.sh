
mkdir -p gliner-multitask-large-v0.5
cd gliner-multitask-large-v0.5
if [ ! -f "tokenizer.json" ]; then
    wget https://huggingface.co/onnx-community/gliner-multitask-large-v0.5/resolve/main/tokenizer.json
fi
mkdir -p onnx
cd onnx
if [ ! -f "model.onnx" ]; then
    wget https://huggingface.co/onnx-community/gliner-multitask-large-v0.5/resolve/main/onnx/model.onnx
fi