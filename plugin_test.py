import ctypes
import tensorrt as trt
import numpy as np
import torch

LIB_PATH = "/workspace/TensorRT/build/out/libnvinfer_plugin.so"
ctypes.CDLL(LIB_PATH)
TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

def get_trt_plugin(plugin_name):
        plugin = None
        for plugin_creator in PLUGIN_CREATORS:
            if plugin_creator.name == plugin_name:
                coefficient_field = trt.PluginField("coefficient", np.array([2], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                field_collection = trt.PluginFieldCollection([coefficient_field])
                plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
        return plugin

# builder = trt.Builder(TRT_LOGGER)
# network = builder.create_network()
# config = builder.create_builder_config()
# config.max_workspace_size = 2**20
# input_layer = network.add_input(name="input_layer", dtype=trt.float32, shape=(1, 1))
# test_plugin = network.add_plugin_v2(inputs=[input_layer], plugin=get_trt_plugin("TEST_TRT"))
# test_plugin.get_output(0).name = "outputs"

# serialized_engine = builder.build_serialized_network(network, config)
# runtime = trt.Runtime(TRT_LOGGER)
# engine = runtime.deserialize_cuda_engine(serialized_engine)

# context = engine.create_execution_context()
# input_idx = engine["input_layer"]
# output_idx = engine["outputs"]

# buffers = [None] * 2 # Assuming 1 input and 1 output

# input_tensor = torch.tensor([2.], device=torch.device('cuda'))
# output_tensor = torch.tensor([None], device=torch.device('cuda'))

# input_ptr = input_tensor.data_ptr()
# output_ptr = output_tensor.data_ptr()   

# buffers[input_idx] = input_ptr
# buffers[output_idx] = output_ptr

# stream = torch.cuda.Stream(device = torch.device('cuda'))
# stream_ptr = stream.cuda_stream

# context.execute_v2(buffers, stream_ptr)
# #context.execute_async_v2(buffers, stream_ptr)
# stream.synchronize()

# print("test complete")


for plugin_creator in PLUGIN_CREATORS:
    print(plugin_creator.name)