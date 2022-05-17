import ctypes
import tensorrt as trt
import numpy as np
import torch

LIB_PATH = "/workspace/TensorRT/testplugin/leakyReluTestPlugin/build/liblrelutest_trt.so"
ctypes.CDLL(LIB_PATH)
TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

for plugin_creator in PLUGIN_CREATORS:
    print(plugin_creator.name)