# Image-Classification-on-modified-ResNet
A custom image classification model based on ResNet architecture, enhanced with self created ResNeXt bottleneck module from improved performance

## Features
- Custom ResNeXt bottleneck for improved classification performance.
- Performance Comparision
- Model export capabilities to TorchScript and ONNX.

## Performance Comparison
The modified ResNet with new Bottleneck module outperformed ResNet
![Loss and Accuracy of ResNet50 vs Modified ResNet50](runs/train/experiment2/loss_acc_plot.png)

## Inference Time Comparison
ONNX inference time is significantly faster
![Model being exported to ONNX vs Model before export to ONNX](ResNet50_inference/inference_time_comparison.png)
