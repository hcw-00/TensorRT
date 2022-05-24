#! /bin/bash

python3 build_engine.py --onnx /workspace/TensorRT/testplugin/leakyReluTestPlugin/gs_test/test_conv_plugin.onnx --engine /workspace/TensorRT/testplugin/leakyReluTestPlugin/gs_test/test_scratch.trt --precision fp16 --calib_batch_size 1