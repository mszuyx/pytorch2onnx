# pytorch2onnx

## pytorch2onnx.py
This script tries to:
- convert a pytorch model to onnx
- check the converted model's inference time
- compare the difference in model results (the onnx model vs. the orignal)
- visualize the results from both models

## onnx_model_shrinker.py
This script is used to shrink the onnx model if it contains shared weights.
